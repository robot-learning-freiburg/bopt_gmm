import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import TanhNormal


LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
MEAN_MIN = -9.0
MEAN_MAX = 9.0


class Actor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_space,
        num_layers: int = 2,
        hidden_dim: int = 256,
        init_w: float = 1e-3,
        device='cuda'
    ):
        super(Actor, self).__init__()
        # Action parameters
        self.action_space = action_space
        action_dim = action_space.shape[0]
        self._device = device

        self.fc_layers = [nn.Linear(input_dim, hidden_dim).to(self._device)]
        for i in range(num_layers - 1):
            self.fc_layers.append(nn.Linear(hidden_dim, hidden_dim).to(self._device))
        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.fc_mean = nn.Linear(hidden_dim, action_dim).to(self._device)
        self.fc_log_std = nn.Linear(hidden_dim, action_dim).to(self._device)
        # https://arxiv.org/pdf/2006.05990.pdf
        # recommends initializing the policy MLP with smaller weights in the last layer
        self.fc_log_std.weight.data.uniform_(-init_w, init_w)
        self.fc_log_std.bias.data.uniform_(-init_w, init_w)
        self.fc_mean.weight.data.uniform_(-init_w, init_w)
        self.fc_mean.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        num_layers = len(self.fc_layers)
        state = F.silu(self.fc_layers[0](state))
        for i in range(1, num_layers):
            state = F.silu(self.fc_layers[i](state))
        mean = self.fc_mean(state)
        mean = torch.clamp(mean, MEAN_MIN, MEAN_MAX)
        log_std = self.fc_log_std(state)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = log_std.exp()
        return mean, std

    def get_actions(
        self,
        state,
        deterministic=False,
        reparameterize=False
    ):
        mean, std = self.forward(state)
        if deterministic:
            actions = torch.tanh(mean)
            log_pi  = torch.zeros_like(actions)
        else:
            tanh_normal = TanhNormal(mean, std)
            if reparameterize:
                actions, log_pi = tanh_normal.rsample_and_logprob()
            else:
                actions, log_pi = tanh_normal.sample_and_logprob()
            return self.scale_actions(actions), log_pi
        return self.scale_actions(actions), log_pi

    def scale_actions(self, action):
        action_high = torch.tensor(self.action_space.high, 
                                   dtype=torch.float, 
                                   device=action.device)
        action_low = torch.tensor(self.action_space.low, 
                                  dtype=torch.float, 
                                  device=action.device)
        slope = (action_high - action_low) / 2
        action = action_low + slope * (action + 1)
        return action


class D2RLActor(Actor):
    def __init__(
        self,
        input_dim: int,
        action_space,
        num_layers: int = 2,
        hidden_dim: int = 256,
        init_w: float = 1e-3,
    ):
        super(D2RLActor, self).__init__(
            input_dim=input_dim,
            action_space=action_space,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            init_w=init_w,
        )
        del self.fc_layers
        aux_dim = input_dim + hidden_dim
        self.fc_layers = [nn.Linear(input_dim, hidden_dim)]
        for i in range(num_layers - 1):
            self.fc_layers.append(nn.Linear(aux_dim, hidden_dim))
        self.fc_layers = nn.ModuleList(self.fc_layers)

    def get_last_hidden_state(self, policy_input):
        num_layers = len(self.fc_layers)
        x = F.silu(self.fc_layers[0](policy_input))
        for i in range(1, num_layers):
            x = torch.cat([x, policy_input], dim=-1)
            x = F.silu(self.fc_layers[i](x))
        return x

    def forward(self, state):
        x = self.get_last_hidden_state(state)
        mean = self.fc_mean(x)
        mean = torch.clamp(mean, MEAN_MIN, MEAN_MAX)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = log_std.exp()
        return mean, std


class DenseNetActor(Actor):
    def __init__(
        self,
        input_dim: int,
        action_space,
        num_layers: int = 2,
        hidden_dim: int = 256,
        init_w: float = 1e-3,
        device='cuda'
    ):
        super(DenseNetActor, self).__init__(
            input_dim=input_dim,
            action_space=action_space,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            init_w=init_w,
            device=device
        )
        action_dim = action_space.shape[0]
        del self.fc_layers
        self.fc_layers = []
        fc_in_features = input_dim
        for _ in range(num_layers):
            self.fc_layers.append(nn.Linear(fc_in_features, hidden_dim).to(device))
            fc_in_features += hidden_dim
        self.fc_layers = nn.ModuleList(self.fc_layers)
        self.fc_mean = nn.Linear(fc_in_features, action_dim).to(device)
        self.fc_log_std = nn.Linear(fc_in_features, action_dim).to(device)
        self.fc_log_std.weight.data.uniform_(-init_w, init_w)
        self.fc_log_std.bias.data.uniform_(-init_w, init_w)
        self.fc_mean.weight.data.uniform_(-init_w, init_w)
        self.fc_mean.bias.data.uniform_(-init_w, init_w)

    def get_last_hidden_state(self, fc_input):
        num_layers = len(self.fc_layers)
        for i in range(num_layers):
            fc_output = F.silu(self.fc_layers[i](fc_input))
            fc_input = torch.cat([fc_input, fc_output], dim=-1)
        return fc_input

    def forward(self, state):
        x = self.get_last_hidden_state(state)
        mean = self.fc_mean(x)
        mean = torch.clamp(mean, MEAN_MIN, MEAN_MAX)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = log_std.exp()
        return mean, std
