from torch import nn
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

# set the QR-DQN
class QRDQN():
    def __init__(self, num_actions, state_dim, agent_id, device, N):
        super(QRDQN, self).__init__()
        hidden_1 = 64
        hidden_2 = 256
        self.sharedlayer = nn.Sequential(
            nn.Linear(state_dim, hidden_1),
            nn.ReLU())
        self.tower = nn.Sequential(
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, num_actions * N))

        self.N = N
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.agent_id = agent_id
        self.lr = 0.001
        #self.opt = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.opt = torch.optim.Adam([{'params': self.sharedlayer.parameters()}, {'params': self.tower.parameters()}],
                                     lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.opt, mode='min', factor=0.1, patience=2)

        self.loss_func = torch.nn.MSELoss()
        self.loss_func = self.loss_func.to(device)

        self.sharedlayer = self.sharedlayer.to(device)
        self.tower = self.tower.to(device)


        self.path = './' + 'params' + '/' + 'agent' + str(agent_id) + '_' + 'params.pth'

    def forward(self, batch_size, states):
        assert states is not None

        h_shared = self.sharedlayer(states)
        qu = self.tower(h_shared)

        quantiles = qu.view(batch_size, self.N, self.num_actions)

        assert quantiles.shape == (batch_size, self.N, self.num_actions)

        #print('quant', quantiles)

        return quantiles

    def calculate_q(self, batch_size, states):
        assert states is not None
        states = torch.unsqueeze(states, 0)

        # Calculate quantiles.
        quantiles = self.forward(batch_size, states)

        # Calculate expectations of value distributions.
        q = quantiles.mean(dim=1)

        assert q.shape == (batch_size, self.num_actions)

        return q

    def save_parameters(self, subpath):
        path1 = self.path + '_' + subpath + '_' + 'shared_net'
        path2 = self.path + '_' + subpath + '_' + 'tower'
        torch.save(self.sharedlayer.state_dict(), path1)
        torch.save(self.tower.state_dict(), path2)
        

    def load_parameters(self, subpath):
        print('load')
        path1 = self.path + '_' + subpath + '_' + 'shared_net'
        path2 = self.path + '_' + subpath + '_' + 'tower'
        
        self.sharedlayer.load_state_dict(torch.load(path1, map_location=torch.device('cpu')))
        self.tower.load_state_dict(torch.load(path2, map_location=torch.device('cpu')))

        # print('parameters', self.params)
        # print('load2')
        # for p in self.sharedlayer.parameters():
        #     print('p', p)

net = QRDQN
print(net)