from torch import nn
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from qrdqn import QRDQN

class IQN(QRDQN):
    def __init__(self, num_actions, state_dim, agent_id, device, N, num_tau_samples, num_tau_prime_samples, kappa):
        super(IQN, self).__init__(num_actions, state_dim, agent_id, device, N)
        self.num_tau_samples = num_tau_samples
        self.num_tau_prime_samples = num_tau_prime_samples
        self.kappa = kappa
    def forward(self, states):
        assert states is not None

        batch_size = states.size(0)

        tau = torch.rand(self.num_tau_samples, device=states.device)
        tau = tau.unsqueeze(0).expand(batch_size, -1)  # [batch_size, num_tau_samples]

        h_shared = self.sharedlayer(states)
        qu = self.tower(h_shared)

        quantiles = qu.view(batch_size, self.N, self.num_actions)

        assert quantiles.shape == (batch_size, self.N, self.num_actions)

        # in IQN, the output is estimated Q value
        q = self.calculate_quantile_huber_loss(quantiles, tau)

        return q

    def calculate_quantile_huber_loss(self, quantiles, tau):
        pairwise_diff = quantiles[:, :, None] - quantiles[:, None, :]  # [N,taus,taus] Shape diff between each tau
        huber_loss = pairwise_diff.clamp(-self.kappa, self.kappa).abs().mean()  # Huber loss elementwise, then averaged

        quantile_huber_loss = (torch.abs(tau[..., None] - (pairwise_diff.detach() < 0).float()) * huber_loss).mean()

        return quantile_huber_loss

net = IQN
print(net)