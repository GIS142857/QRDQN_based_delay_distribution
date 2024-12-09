import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.distribution import Categorical
import numpy as np
import random
import gym
import math
from collections import deque
from visualdl import LogWriter
from tqdm import trange


def set_seed(seed):
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class replay_buffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    def store(self, observation, action, reward, next_observation, done):
        observation = np.expand_dims(observation, 0)
        next_observation = np.expand_dims(next_observation, 0)
        self.memory.append([observation, action, reward, next_observation, done])

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        observation, action, reward, next_observation, done = zip(*batch)
        return np.concatenate(observation, 0), action, reward, np.concatenate(next_observation, 0), done

    def __len__(self):
        return len(self.memory)


class fqf_net(nn.Layer):
    def __init__(self, observation_dim, action_dim, quant_num, cosine_num):
        super(fqf_net, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.quant_num = quant_num
        self.cosine_num = cosine_num

        self.feature_layer = nn.Sequential(
            nn.Linear(self.observation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.cosine_layer = nn.Sequential(
            nn.Linear(self.cosine_num, 128),
            nn.ReLU()
        )
        self.psi_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim)
        )
        self.quantile_fraction_layer = nn.Sequential(
            nn.Linear(128, self.quant_num),
            nn.Softmax(axis=-1)
        )

    def calc_state_embedding(self, observation):
        return self.feature_layer(observation)

    def calc_quantile_fraction(self, state_embedding):
        assert state_embedding.stop_gradient
        q = self.quantile_fraction_layer(state_embedding.detach())
        tau_0 = paddle.zeros([q.shape[0], 1], dtype="float32")
        tau = paddle.concat([tau_0, q], axis=-1)
        tau = paddle.cumsum(tau, axis=-1)
        entropy = Categorical(q).entropy()
        tau_hat = ((tau[:, :-1] + tau[:, 1:]) / 2.).detach()
        return tau, tau_hat, entropy

    def calc_quantile_value(self, tau, state_embedding):
        assert tau.stop_gradient
        quants = paddle.arange(0, self.cosine_num, 1.0).unsqueeze(0).unsqueeze(0)
        cos_trans = paddle.cos(quants * tau.unsqueeze(-1).detach() * np.pi)
        rand_feat = self.cosine_layer(cos_trans)
        x = state_embedding.unsqueeze(1)
        x = x * rand_feat
        value = self.psi_layer(x).transpose([0, 2, 1])

        return value

    def act(self, observation, epsilon):
        if random.random() > epsilon:
            state_embedding = self.calc_state_embedding(observation)
            tau, tau_hat, _ = self.calc_quantile_fraction(state_embedding.detach())
            q_value = self.calc_q_value(state_embedding, tau, tau_hat)
            action = q_value.argmax().detach().numpy().item()
        else:
            action = random.choice(list(range(self.action_dim)))
        return action

    def calc_sa_quantile_value(self, state_embedding, action, tau):
        sa_quantile_value = self.calc_quantile_value(tau.detach(), state_embedding)
        _sa_quantile_value = sa_quantile_value.gather(action, axis=1)
        eyes = np.eye(_sa_quantile_value.shape[0], _sa_quantile_value.shape[1]).astype("float32")
        eyes = np.repeat(eyes, _sa_quantile_value.shape[-1]).reshape(-1, _sa_quantile_value.shape[1], _sa_quantile_value.shape[-1])
        eyes = paddle.to_tensor(eyes)
        sa_quantile_value = _sa_quantile_value.multiply(eyes).sum(1)
        return sa_quantile_value

    def calc_q_value(self, state_embedding, tau, tau_hat):
        tau_delta = tau[:, 1:] - tau[:, :-1]
        tau_hat_value = self.calc_quantile_value(tau_hat.detach(), state_embedding)
        q_value = (tau_delta.unsqueeze(1) * tau_hat_value).sum(-1).detach()
        return q_value


class fqf(object):
    def __init__(self, env, capacity,
                 episode, exploration,
                 k, gamma, quant_num,
                 cosine_num, batch_size,
                 value_learning_rate, fraction_learning_rate,
                 entropy_weight, epsilon_init,
                 double_q, decay,
                 epsilon_min, update_freq,
                 render, log_dir='./log'):
        self.env = env
        self.capacity = capacity
        self.episode = episode
        self.exploration = exploration
        self.k = k
        self.gamma = gamma
        self.batch_size = batch_size
        self.value_learning_rate = value_learning_rate
        self.fraction_learning_rate = fraction_learning_rate
        self.epsilon_init = epsilon_init
        self.decay = decay
        self.quant_num = quant_num
        self.epsilon_min = epsilon_min
        self.entropy_weight = entropy_weight
        self.update_freq = update_freq
        self.render = render
        self.cosine_num = cosine_num
        self.double_q = double_q

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.net = fqf_net(self.observation_dim, self.action_dim, self.quant_num, self.cosine_num)
        print(self.net)
        self.target_net = fqf_net(self.observation_dim, self.action_dim, self.quant_num, self.cosine_num)
        self.target_net.load_dict(self.net.state_dict())
        self.buffer = replay_buffer(self.capacity)
        self.quantile_value_param = list(self.net.feature_layer.parameters()) + list(
            self.net.cosine_layer.parameters()) + list(self.net.psi_layer.parameters())
        self.quantile_fraction_param = list(self.net.quantile_fraction_layer.parameters())
        self.quantile_value_optimizer = paddle.optimizer.Adam(parameters=self.quantile_value_param,
                                                              learning_rate=self.value_learning_rate)
        self.quantile_fraction_optimizer = paddle.optimizer.RMSProp(parameters=self.quantile_fraction_param,
                                                                    learning_rate=self.fraction_learning_rate)
        self.epsilon = lambda x: self.epsilon_min + (self.epsilon_init - self.epsilon_min) * math.exp(
            -1. * x / self.decay)
        self.count = 0
        self.weight_reward = None
        self.log_dir = log_dir

    def calc_quantile_value_loss(self, tau, value, target_value):
        # 计算 quantile value loss
        # 得到 quantile huber loss
        assert tau.stop_gradient
        u = target_value.unsqueeze(1) - value.unsqueeze(-1)
        huber_loss = 0.5 * u.abs().clip(min=0., max=self.k).pow(2)
        huber_loss = huber_loss + self.k * (u.abs() - u.abs().clip(min=0., max=self.k) - 0.5 * self.k)
        quantile_loss = (tau.unsqueeze(-1) - (u < 0).cast("float32")).abs() * huber_loss
        loss = quantile_loss.mean()
        return loss

    def calc_quantile_fraction_loss(self, observations, actions, tau, tau_hat):
        # 计算 quantile fraction loss
        assert tau_hat.stop_gradient
        sa_quantile_hat = self.net.calc_sa_quantile_value(observations, actions, tau_hat).detach()
        sa_quantile = self.net.calc_sa_quantile_value(observations, actions, tau[:, 1:-1]).detach()
        value_1 = sa_quantile - sa_quantile_hat[:, :-1]
        signs_1 = sa_quantile > paddle.concat([sa_quantile_hat[:, :1], sa_quantile[:, :-1]], axis=-1)
        value_2 = sa_quantile - sa_quantile_hat[:, 1:]
        signs_2 = sa_quantile < paddle.concat([sa_quantile[:, 1:], sa_quantile_hat[:, -1:]], axis=-1)
        gradient_tau_1 = paddle.where(signs_1, value_1, -value_1)
        gradient_tau_2 = paddle.where(signs_2, value_2, -value_2)
        gradient_tau = (gradient_tau_1 + gradient_tau_2).reshape(value_1.shape)
        return (gradient_tau.detach() * tau[:, 1: -1]).sum(1).mean()

    def train(self):
        observations, actions, rewards, next_observations, dones = self.buffer.sample(self.batch_size)

        observations = paddle.to_tensor(observations, dtype="float32")
        actions = paddle.to_tensor(actions, dtype="int64")
        rewards = paddle.to_tensor(rewards, dtype="float32").unsqueeze(1)
        next_observations = paddle.to_tensor(next_observations, dtype="float32")
        dones = paddle.to_tensor(dones, dtype="float32").unsqueeze(1)

        state_embedding = self.net.calc_state_embedding(observations)
        tau, tau_hat, entropy = self.net.calc_quantile_fraction(state_embedding.detach())
        # 计算 quantile value
        dist = self.net.calc_quantile_value(tau_hat.detach(), state_embedding)

        _dist = dist.gather(actions, axis=1)
        eyes = np.eye(_dist.shape[0], _dist.shape[1]).astype("float32")
        eyes = np.repeat(eyes, _dist.shape[-1]).reshape(-1, _dist.shape[1], _dist.shape[-1])
        eyes = paddle.to_tensor(eyes)
        value = _dist.multiply(eyes).sum(1)

        if not self.double_q:
            next_state_embedding = self.target_net.calc_state_embedding(next_observations)
            next_tau, next_tau_hat, _ = self.net.calc_quantile_fraction(next_state_embedding.detach())
            target_actions = self.target_net.calc_q_value(next_state_embedding, next_tau, next_tau_hat).argmax(
                1).detach()
        else:
            next_state_embedding = self.net.calc_state_embedding(next_observations)
            next_tau, next_tau_hat, _ = self.net.calc_quantile_fraction(next_state_embedding.detach())
            target_actions = self.net.calc_q_value(next_state_embedding, next_tau, next_tau_hat).argmax(1).detach()
        next_state_embedding = self.target_net.calc_state_embedding(next_observations)

        target_dist = self.target_net.calc_quantile_value(tau_hat.detach(), next_state_embedding)

        _target_dist = target_dist.gather(target_actions, axis=1)
        eyes = np.eye(_target_dist.shape[0], _target_dist.shape[1]).astype("float32")
        eyes = np.repeat(eyes, _target_dist.shape[-1]).reshape(-1, _target_dist.shape[1], _target_dist.shape[-1])
        eyes = paddle.to_tensor(eyes)
        target_value = _target_dist.multiply(eyes).sum(1)

        target_value = rewards + self.gamma * target_value * (1. - dones)
        target_value = target_value.detach()

        qauntile_value_loss = self.calc_quantile_value_loss(tau_hat.detach(), value, target_value)
        quantile_fraction_loss = self.calc_quantile_fraction_loss(state_embedding, actions, tau, tau_hat)
        entropy_loss = - (self.entropy_weight * entropy).mean()

        self.quantile_fraction_optimizer.clear_grad()
        quantile_fraction_loss.backward(retain_graph=True)
        self.quantile_fraction_optimizer.step()

        self.quantile_value_optimizer.clear_grad()
        qauntile_value_loss.backward()
        self.quantile_value_optimizer.step()

        if self.count % self.update_freq == 0:
            self.target_net.load_dict(self.net.state_dict())

    def main(self):
        log_writer = LogWriter(logdir=self.log_dir, comment="Categorical DQN")
        for i in trange(self.episode):
            obs = self.env.reset()
            if self.render:
                self.env.render()
            total_reward = 0
            while True:
                epsilon = self.epsilon(self.count)
                action = self.net.act(paddle.to_tensor(np.expand_dims(obs, 0), dtype="float32"), epsilon)
                next_obs, reward, done, info = self.env.step(action)
                self.count += 1
                total_reward += reward
                if self.render:
                    self.env.render()
                self.buffer.store(obs, action, reward, next_obs, done)
                obs = next_obs

                if self.count > self.exploration:
                    self.train()

                if done:
                    if not self.weight_reward:
                        self.weight_reward = total_reward
                    else:
                        self.weight_reward = 0.99 * self.weight_reward + 0.01 * total_reward
                    # print('episode: {}  reward: {}  weight_reward: {:.2f}  epsilon: {:.2f}'.format(i + 1, total_reward, self.weight_reward, epsilon))
                    log_writer.add_scalar("Reward", value=paddle.to_tensor(total_reward), step=i + 1)
                    log_writer.add_scalar("Weight Reward", value=paddle.to_tensor(self.weight_reward), step=i + 1)
                    log_writer.add_scalar("epsilon", value=paddle.to_tensor(epsilon), step=i + 1)
                    break

if __name__ == '__main__':
    seed = 777
    set_seed(seed)
    env = gym.make('CartPole-v1')
    env.reset(seed=seed)
    env = env.unwrapped
    test = fqf(
        env=env,
        capacity=10000,
        episode=5000,
        exploration=1000,
        k=1.,
        gamma=0.99,
        batch_size=32,
        quant_num=32,
        cosine_num=64,
        value_learning_rate=1e-3,
        fraction_learning_rate=1e-9,
        entropy_weight=0,
        double_q=True,
        epsilon_init=1,
        decay=5000,
        epsilon_min=0.01,
        update_freq=200,
        render=False,
    )
    test.main()