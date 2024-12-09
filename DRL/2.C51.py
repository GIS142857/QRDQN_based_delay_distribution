from typing import Dict, List, Tuple
import gym
from visualdl import LogWriter
from tqdm import tqdm,trange
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.optimizer as optimizer


def index_add_(parent, axis, idx, child):
    expend_dim = parent.shape[0]
    idx_one_hot = F.one_hot(idx.cast("int64"), expend_dim)
    child = paddle.expand_as(child.cast("float32").unsqueeze(-1), idx_one_hot)
    output = parent + (idx_one_hot.cast("float32").multiply(child)).sum(axis).squeeze()
    return output

class ReplayBuffer:
    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self):
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self):
        return self.size


class C51DQN(nn.Layer):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            atom_size: int,
            support
    ):
        # 初始化
        super(C51DQN, self).__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim * atom_size)
        )

    def forward(self, x):
        dist = self.dist(x)
        q = paddle.sum(dist * self.support, axis=2)
        return q

    def dist(self, x):
        q_atoms = self.layers(x).reshape([-1, self.out_dim, self.atom_size])
        dist = F.softmax(q_atoms, axis=-1)
        dist = dist.clip(min=float(1e-3))  # 避免 nan
        return dist


class C51Agent:
    def __init__(
            self,
            env: gym.Env,
            memory_size: int,
            batch_size: int,
            target_update: int,
            epsilon_decay: float,
            max_epsilon: float = 1.0,
            min_epsilon: float = 0.1,
            gamma: float = 0.99,
            # C51 算法的参数
            v_min: float = 0.0,
            v_max: float = 200.0,
            atom_size: int = 51,
            log_dir: str = "./log"
    ):
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self.env = env
        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma

        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = paddle.linspace(
            self.v_min, self.v_max, self.atom_size
        )

        # 定义网络 两个网络，一个eval网络和target网络
        self.dqn = C51DQN(
            obs_dim, action_dim, atom_size, self.support
        )
        self.dqn_target = C51DQN(
            obs_dim, action_dim, atom_size, self.support
        )
        self.dqn_target.load_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        self.optimizer = optimizer.Adam(parameters=self.dqn.parameters())

        self.transition = []

        self.is_test = False

        self.log_dir = log_dir

        self.log_writer = LogWriter(logdir=self.log_dir, comment="Categorical DQN")

    def select_action(self, state: np.ndarray):
        if self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()
        else:
            selected_action = self.dqn(
                paddle.to_tensor(state, dtype="float32"),
            ).argmax()
            selected_action = selected_action.detach().numpy()

        if not self.is_test:
            self.transition = [state, selected_action]

        return selected_action

    def step(self, action: np.ndarray):
        next_state, reward, done, _ = self.env.step(int(action))

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)

        return next_state, reward, done

    def update_model(self):
        samples = self.memory.sample_batch()

        loss = self._compute_dqn_loss(samples)

        self.optimizer.clear_grad()
        loss.backward()
        self.optimizer.step()
        loss_show = loss
        return loss_show.numpy().item()

    def train(self, num_frames: int, plotting_interval: int = 200):
        self.is_test = False

        state = self.env.reset()
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0
        epsilon = 0

        for frame_idx in trange(1, num_frames + 1):
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            # 回合结束
            if done:
                epsilon += 1
                state = self.env.reset()
                self.log_writer.add_scalar("Reward", value=paddle.to_tensor(score), step=epsilon)
                scores.append(score)
                score = 0

            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                self.log_writer.add_scalar("Loss", value=paddle.to_tensor(loss), step=frame_idx)
                losses.append(loss)
                update_cnt += 1

                self.epsilon = max(
                    self.min_epsilon, self.epsilon - (
                            self.max_epsilon - self.min_epsilon
                    ) * self.epsilon_decay
                )
                epsilons.append(self.epsilon)

                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

        self.env.close()

    def test(self):
        self.is_test = True
        state = self.env.reset()
        done = False
        score = 0
        frames = []
        while not done:
            frames.append(self.env.render(mode="rgb_array"))
            action = self.select_action(state)
            next_state, reward, done = self.step(int(action))
            state = next_state
            score += reward
        print("score: ", score)
        self.env.close()
        return frames

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]):
        # 计算损失
        state = paddle.to_tensor(samples["obs"], dtype="float32")
        next_state = paddle.to_tensor(samples["next_obs"], dtype="float32")
        action = paddle.to_tensor(samples["acts"], dtype="int64")
        reward = paddle.to_tensor(samples["rews"].reshape([-1, 1]), dtype="float32")
        done = paddle.to_tensor(samples["done"].reshape([-1, 1]), dtype="float32")

        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with paddle.no_grad():
            next_action = self.dqn_target(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[:self.batch_size, ]
            _next_dist = paddle.gather(next_dist, next_action, axis=1)
            eyes = np.eye(_next_dist.shape[0], _next_dist.shape[1]).astype("float32")
            eyes = np.repeat(eyes, _next_dist.shape[-1]).reshape(-1, _next_dist.shape[1], _next_dist.shape[-1])
            eyes = paddle.to_tensor(eyes)

            next_dist = _next_dist.multiply(eyes).sum(1)

            t_z = reward + (1 - done) * self.gamma * self.support
            t_z = t_z.clip(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().cast("int64")
            u = b.ceil().cast("int64")

            offset = (
                paddle.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).cast("int64")
                .unsqueeze(1)
                .expand([self.batch_size, self.atom_size])
            )

            proj_dist = paddle.zeros(next_dist.shape)
            proj_dist = index_add_(
                proj_dist.reshape([-1]),
                0,
                (l + offset).reshape([-1]),
                (next_dist * (u.cast("float32") - b)).reshape([-1])
            )
            proj_dist = index_add_(
                proj_dist.reshape([-1]),
                0,
                (u + offset).reshape([-1]),
                (next_dist * (b - l.cast("float32"))).reshape([-1])
            )
            proj_dist = proj_dist.reshape(next_dist.shape)

        dist = self.dqn.dist(state)
        _dist = paddle.gather(dist[:self.batch_size, ], action, axis=1)
        eyes = np.eye(_dist.shape[0], _dist.shape[1]).astype("float32")
        eyes = np.repeat(eyes, _dist.shape[-1]).reshape(-1, _dist.shape[1], _dist.shape[-1])
        eyes = paddle.to_tensor(eyes)
        dist_batch = _dist.multiply(eyes).sum(1)
        log_p = paddle.log(dist_batch)

        loss = -(proj_dist * log_p).sum(1).mean()

        return loss

    def _target_hard_update(self):
        # 更新目标模型参数
        self.dqn_target.load_dict(self.dqn.state_dict())



env_id = "CartPole-v1"
env = gym.make(env_id)

seed = 777

np.random.seed(seed)
paddle.seed(seed)
env.reset(seed=seed)

num_frames = 30000
memory_size = 1000
batch_size = 32
target_update = 200
epsilon_decay = 1 / 2000

# 训练
agent = C51Agent(env, memory_size, batch_size, target_update, epsilon_decay)
agent.train(num_frames)