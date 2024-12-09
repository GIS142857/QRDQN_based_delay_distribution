import torch
from torch.distributions import Categorical
from torch import nn
from qrdqn import QRDQN
class FQF(QRDQN):
    def __init__(self, num_actions, state_dim, agent_id, device, N):
        super(FQF, self).__init__(num_actions, state_dim, agent_id, device, N)

        self.pi_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.N),
            nn.Softmax(dim=-1))
        self.pi_net.to(device)
    def forward(self, batch_size, states):
        assert states is not None

        h_shared = self.sharedlayer(states)
        qu = self.tower(h_shared)

        logits = self.pi_net(states)
        dist = Categorical(logits)
        tau = dist.sample([self.N])
        tau.sort(dim=1)

        quantiles = qu.view(batch_size, self.N, self.num_actions)

        assert quantiles.shape == (batch_size, self.N, self.num_actions)

        return quantiles, tau

    def calculate_q(self, batch_size, states):
        assert states is not None
        states = torch.unsqueeze(states, 0)

        # Calculate quantiles.
        quantiles, tau = self.forward(batch_size, states)

        # Calculate expectations of value distributions.
        q = (quantiles * tau.unsqueeze(-1)).sum(dim=1)

        assert q.shape == (batch_size, self.num_actions)

        return q

    def save_parameters(self, subpath):
        super().save_parameters(subpath)
        path = self.path + '_' + subpath + '_' + 'pi_net'
        torch.save(self.pi_net.state_dict(), path)

    def load_parameters(self, subpath):
        super().load_parameters(subpath)
        print('load pi_net')
        path = self.path + '_' + subpath + '_' + 'pi_net'
        self.pi_net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

net = FQF
print(net)
