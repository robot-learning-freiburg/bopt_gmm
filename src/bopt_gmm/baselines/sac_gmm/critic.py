import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_layers: int = 2,
        hidden_dim: int = 256,
        device='cuda'
    ):
        super(Critic, self).__init__()
        self.Q1 = QNetwork(input_dim=input_dim, 
                           num_layers=num_layers, 
                           hidden_dim=hidden_dim, 
                           device=device)
        self.Q2 = QNetwork(input_dim=input_dim, 
                           num_layers=num_layers, 
                           hidden_dim=hidden_dim, 
                           device=device)

    def forward(self, state, action):
        q1 = self.Q1(state, action)
        q2 = self.Q2(state, action)
        return q1, q2


class QNetwork(nn.Module):
    def __init__(self, input_dim: int, num_layers: int = 2, hidden_dim: int = 256, device='cuda'):
        super(QNetwork, self).__init__()
        self._device   = device
        self.fc_layers = [nn.Linear(input_dim, hidden_dim).to(self._device)]
        for i in range(num_layers - 1):
            self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim).to(self._device))
        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.out = nn.Linear(hidden_dim, 1).to(self._device)

    def forward(self, state, action):
        fc_input = torch.cat((state, action), dim=-1)
        num_layers = len(self.fc_layers)
        x = F.silu(self.fc_layers[0](fc_input))
        for i in range(1, num_layers):
            x = F.silu(self.fc_layers[i](x))
        return self.out(x)


class D2RLCritic(Critic):
    def __init__(
        self,
        input_dim: int,
        num_layers: int = 2,
        hidden_dim: int = 256,
    ):
        super(D2RLCritic, self).__init__(
            input_dim=input_dim,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
        )
        self.Q1 = D2RLQNetwork(
            input_dim=input_dim, num_layers=num_layers, hidden_dim=hidden_dim
        )
        self.Q2 = D2RLQNetwork(
            input_dim=input_dim, num_layers=num_layers, hidden_dim=hidden_dim
        )


class D2RLQNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_layers: int = 2,
        hidden_dim: int = 256,
    ):
        super(D2RLQNetwork, self).__init__()
        aux_dim = input_dim + hidden_dim
        self.fc_layers = [nn.Linear(input_dim, hidden_dim)]
        for i in range(num_layers - 1):
            self.fc_layers.append(nn.Linear(aux_dim, hidden_dim))
        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        q_input = torch.cat((state, action), dim=-1)
        num_layers = len(self.fc_layers)
        x = F.silu(self.fc_layers[0](q_input))
        for i in range(1, num_layers):
            x = torch.cat([x, q_input], dim=-1)
            x = F.silu(self.fc_layers[i](x))
        return self.out(x)


class DenseNetCritic(Critic):
    def __init__(self, input_dim: int,
                       num_layers: int = 2,
                       hidden_dim: int = 256,
                       device='cuda'):
        super(DenseNetCritic, self).__init__(input_dim=input_dim,
                                             num_layers=num_layers,
                                             hidden_dim=hidden_dim,
                                             device=device)
        self.Q1 = DenseNetQNetwork(input_dim=input_dim, 
                                   num_layers=num_layers, 
                                   hidden_dim=hidden_dim,
                                   device=device)
        
        self.Q2 = DenseNetQNetwork(input_dim=input_dim, 
                                   num_layers=num_layers, 
                                   hidden_dim=hidden_dim,
                                   device=device)


class DenseNetQNetwork(nn.Module):
    def __init__(self, input_dim: int,
                       num_layers: int = 2,
                       hidden_dim: int = 256,
                       device='cuda'):
        super(DenseNetQNetwork, self).__init__()
        self._device = device
        fc_in_features = input_dim
        self.fc_layers = []
        for _ in range(num_layers):
            self.fc_layers.append(nn.Linear(fc_in_features, hidden_dim).to(self._device))
            fc_in_features += hidden_dim
        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.out = nn.Linear(fc_in_features, 1).to(self._device)

    def forward(self, state, action):
        fc_input = torch.cat((state, action), dim=-1)
        num_layers = len(self.fc_layers)
        for i in range(num_layers):
            fc_output = F.silu(self.fc_layers[i](fc_input))
            fc_input = torch.cat([fc_input, fc_output], dim=-1)
        return self.out(fc_input)
