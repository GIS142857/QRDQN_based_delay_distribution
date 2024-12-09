import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# 创建一个神经网络模型
class QRDQN(nn.Module):
    def __init__(self, n_actions, n_quantiles):
        super(QRDQN, self).__init__()
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles

        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, n_actions * n_quantiles)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, self.n_actions, self.n_quantiles)

# 创建一个经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# 定义 QR-DQN 代理
class QRDQNAgent:
    def __init__(self, n_actions, n_quantiles, epsilon_start, epsilon_end, epsilon_decay, target_update):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update

        self.policy_net = QRDQN(n_actions, n_quantiles).to(self.device)
        self.target_net = QRDQN(n_actions, n_quantiles).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayBuffer(10000)

        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()
        self.epsilon = self.epsilon_end + (1.0 - self.epsilon_end) * np.exp(-1.0 * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        if sample > self.epsilon:
            with torch.no_grad():
                state = torch.tensor([state], device=self.device, dtype=torch.float32)
                quantiles = self.policy_net(state)
                values = quantiles.mean(2)
                action = values.max(1)[1].view(1, 1)
                return action.item()
        else:
            return random.randrange(self.n_actions)

    def optimize_model(self, batch_size, gamma, kappa):
        if len(self.memory) < batch_size:
            return

        transitions = self.memory.sample(batch_size)
        batch = list(zip(*transitions))
        state_batch = torch.tensor(batch[0], device=self.device, dtype=torch.float32)
        action_batch = torch.tensor(batch[1], device=self.device, dtype=torch.long)
        reward_batch = torch.tensor(batch[2], device=self.device, dtype=torch.float32)
        next_state_batch = torch.tensor(batch[3], device=self.device, dtype=torch.float32)
        done_batch = torch.tensor(batch[4], device=self.device, dtype=torch.uint8)

        quantiles = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(2).expand(batch_size, 1, self.n_quantiles))

        next_quantiles = self.target_net(next_state_batch).detach()
        next_actions = next_quantiles.mean(2).max(1)[1]
        next_quantiles = next_quantiles[range(batch_size), next_actions]

        target = reward_batch + gamma * next_quantiles * (1 - done_batch)

        quantile_huber_loss = self.quantile_huber_loss(quantiles, target.unsqueeze(1), kappa)
        loss = quantile_huber_loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def quantile_huber_loss(self, quantiles, target, kappa):
        u = target.unsqueeze(2) - quantiles
        huber_loss = 0.5 * u.abs().clamp(min=0, max=kappa).pow(2)
        delta = (u < 0).float()
        loss = (delta * kappa * huber_loss + (1 - delta) * huber_loss).mean(2)
        return loss

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# 设置超参数
n_actions = 2
n_quantiles = 51
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 500
target_update = 10
gamma = 0.99
kappa = 1.0
batch_size = 64
lr = 0.001
num_episodes = 1000

# 初始化 QR-DQN 代理
agent = QRDQNAgent(n_actions, n_quantiles, epsilon_start, epsilon_end, epsilon_decay, target_update)

# 创建环境
env = gym.make('CartPole-v1')

# 训练 QR-DQN 代理
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    for t in range(1000):  # 假设每个 episode 最多运行 1000 步
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.memory.push(state, action, reward, next_state, done)
        state = next_state

        total_reward += reward

        agent.optimize_model(batch_size, gamma, kappa)

        if done:
            break

    if episode % target_update == 0:
        agent.update_target()

    if episode % 10 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

# 关闭环境
env.close()
