import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses                import dataclass
from gym.spaces                 import Box
from torch.distributions.normal import Normal

from .mlp import MLP


@dataclass
class LSTMPolicyConfig:
    learning_rate : float
    weight_decay  : float

    def serialize(self):
        return {'learning_rate': self.learning_rate,
                'weight_decay' : self.weight_decay}


class LSTMPolicy(nn.Module):
    def __init__(self, embedding_mlp : MLP, 
                       action_dim : int, 
                       config : LSTMPolicyConfig,
                       lstm_params=None,
                       action_params=None,
                       device='cuda:0'):
        super().__init__()
        
        self.mlp     = embedding_mlp
        self._ad     = action_dim
        self._device = device
        self._config = config

        self.lstm    = nn.LSTM(self.mlp.output_dim, self.mlp.output_dim).to(device)  # , batch_first=True)
        if lstm_params is not None:
            self.lstm.load_state_dict(lstm_params)

        self.action_head = nn.Linear(self.mlp.output_dim, action_dim).to(device)
        if action_params is not None:
            self.action_head.load_state_dict(action_params)

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.std = 0.1 * torch.ones(action_dim, dtype=torch.float32)
        self.std = self.std.to(self._device)
        return

    @property
    def device(self):
        return self._device

    def forward_step(self, observation, lstm_state):
        embedding = self.mlp(observation)
        lstm_out, lstm_state = self.lstm(embedding, lstm_state)
        out = torch.tanh(self.action_head(lstm_out))
        return out, lstm_state

    def forward(self, observation_traj, action_traj):
        losses = []
        lstm_state = None
        for observation, action in zip(observation_traj, action_traj):
            mu, lstm_state = self.forward_step(observation, lstm_state)
            distribution   = Normal(mu, self.std)
            log_prob       = distribution.log_prob(action)
            losses.append(-log_prob)

        return torch.cat(losses).mean()

    def update_params(self, observation_traj, action_traj):
        observations = observation_traj.to(self._device)
        actions      = action_traj.to(self._device)
        self.optimizer.zero_grad()
        loss = self.forward(observations, actions)
        loss.backward()
        self.optimizer.step()
        return {"loss": loss}

    def predict(self, observation, lstm_state):
        observation_th = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self._device)
        
        with torch.no_grad():
            action_th, lstm_state = self.forward_step(observation_th, lstm_state)
            action = action_th.detach().cpu().squeeze(0).squeeze(0).numpy()
        return action, lstm_state

    @classmethod
    def deserialize(cls, d, device='cuda:0'):
        d['embedding_mlp'] = MLP.deserialize(d['embedding_mlp'], device=device)
        d['config']        = LSTMPolicyConfig(**d['config'])
        return cls(**d, device=device)

    def serialize(self):
        return {'embedding_mlp' : self.mlp.serialize(),
                'action_dim'    : self._ad,
                'config'        : self._config.serialize(),
                'action_params' : self.action_head.state_dict(),
                'lstm_params'   : self.lstm.state_dict()}
    
    @classmethod
    def load_model(cls, path, device='cuda:0'):
        return cls.deserialize(torch.load(path), device=device)

    def save_model(self, path):
        torch.save(self.serialize(), path)
