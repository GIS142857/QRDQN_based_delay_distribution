import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class IQNNet(nn.Module):
    def __init__(self, num_states, nums_tao, num_actions, num_quantiles, embedding_dim=64):
        super(IQNNet, self).__init__()
        self.num_states = num_states
        self.nums_tao = nums_tao
        self.num_actions = num_actions
        self.num_quantiles = num_quantiles
        self.embedding_dim = embedding_dim
        # 状态特征提取
        self.feature_layer = nn.Sequential(nn.Linear(num_states, embedding_dim),nn.ReLU())
        # 分位数嵌入层
        self.quantile_embedding_layer = nn.Linear(nums_tao, embedding_dim)
        # 输出层
        self.value_layer = nn.Linear(embedding_dim, num_actions*num_quantiles)

    def forward(self, state, quantiles):
        # 状态编码
        state_embed = self.feature_layer(state)

        # 计算余弦嵌入
        batch_size = state.size(0)
        quantile_embed = torch.cos(
            torch.arange(1, self.embedding_dim + 1, 1).float().unsqueeze(0).unsqueeze(0) *
            quantiles.unsqueeze(-1) * math.pi
        ).to(state.device)

        # 计算量子嵌入
        quantile_embed = self.quantile_embedding_layer(quantile_embed).view(batch_size, -1, self.embedding_dim)

        # 将状态嵌入与量子嵌入结合
        combined = F.relu(quantile_embed * state_embed.unsqueeze(1))

        # 计算最终动作值预测
        return self.value_layer(combined).view(batch_size, self.num_quantiles, self.num_actions)


# 实例化网络和采样τ值
num_states = ...  # 状态空间维度
num_actions = ...  # 动作空间维度
num_quantiles = 8  # 分位数的数量
embedding_dim = 64  # 嵌入维度

iqn_net = IQNNet(num_states, num_actions, num_quantiles, embedding_dim)

# 示例状态
state = torch.rand(1, num_states)

# 从U(0,1)中采样τ值
quantiles = torch.rand(1, num_quantiles)

# 计算网络输出
action_values = iqn_net(state, quantiles)
