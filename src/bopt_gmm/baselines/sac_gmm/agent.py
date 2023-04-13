import numpy as np
import torch
import torch.nn            as nn
import torch.nn.functional as F
import torch.optim         as optim

from dataclasses    import dataclass
from tqdm           import tqdm
from typing         import Tuple

from .replay_buffer import ReplayBuffer


@dataclass
class SACConfig:
    gamma      = 0.99
    tau        = 0.005
    lr_actor   = 3e-4
    lr_critic  = 3e-4
    lr_alpha   = 3e-4
    init_alpha = 0.01
    batch_size = 256
    warm_start_steps   = 256
    fill_strategy      = 'random'
    replay_buffer_size = 100000
    replay_buffer_path = None


class SACAgent(object):
    """
    Base Agent class handling the interaction with the environment
    """

    def __init__(self, config : SACConfig,
                       actor_network,
                       critic_network,
                       critic_target_network,
                       action_space,
                       logger = None,
                       device='cuda') -> None:
        """
        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
        """
        self.config        = config
        self.actor         = actor_network
        self.critic        = critic_network
        self.critic_target = critic_target_network
        self.logger        = logger
        self._action_space = action_space
        self._device       = device
        self.replay_buffer = ReplayBuffer(self.config.replay_buffer_size)

        self.target_entropy = -np.prod(action_space.shape).item() 

        self.log_alpha  = nn.Parameter(torch.Tensor([np.log(self.config.init_alpha)]).to(self._device), 
                                       requires_grad=True)

        self.opt_critic = optim.Adam(filter(lambda p: p.requires_grad, self.critic.parameters()),
                                     lr=self.config.lr_critic)
        self.opt_actor  = optim.Adam(filter(lambda p: p.requires_grad, self.actor.parameters()), 
                                    lr=self.config.lr_actor)
        self.opt_alpha  = optim.Adam([self.log_alpha], lr=self.config.lr_alpha)


    @property
    def metrics(self):
        return {'alpha', 'critic_loss', 'actor_loss', 'alpha_loss'}

    @property
    def warm_start_steps(self):
        return max(self.config.warm_start_steps, self.config.batch_size)

    @property
    def action_space(self):
        return self._action_space

    def append_to_replay_buffer(self, prior_obs, action, post_obs, reward, done):
        self.replay_buffer.add_transition(prior_obs, action, post_obs, reward, done)

    def get_action(self, observations, strategy="stochastic", device="cuda"):
        """Interface to get action from SAC Actor,
        ready to be used in the environment"""
        observations = torch.tensor(observations, dtype=torch.float32).to(self._device)
        if strategy == "stochastic":
            action, _ = self.actor.get_actions(observations, 
                                               deterministic=False, 
                                               reparameterize=False)
            return action.detach().cpu().numpy()
        elif strategy == "deterministic":
            action, _ = self.actor.get_actions(observations,
                                               deterministic=True,
                                               reparameterize=False)
            return action.detach().cpu().numpy()
        elif strategy == "random":
            return self.action_space.sample()
        elif strategy == "zeros":
            return np.zeros(self.action_space.shape)
        else:
            raise Exception("Strategy not implemented")
        
    def compute_critic_loss(self, batch):
        batch_observations,      \
        batch_actions,           \
        batch_next_observations, \
        batch_rewards,           \
        batch_dones = batch

        batch_observations      = torch.tensor(batch_observations, dtype=torch.float32).to(self._device)
        batch_actions           = torch.tensor(batch_actions, dtype=torch.float32).to(self._device)
        batch_next_observations = torch.tensor(batch_next_observations, dtype=torch.float32).to(self._device)
        # Need to unsqueeze as they have no batch dimension by default
        batch_rewards           = torch.tensor(batch_rewards, dtype=torch.float32).to(self._device).unsqueeze(1)
        batch_dones             = torch.tensor(batch_dones, dtype=torch.float32).to(self._device).unsqueeze(1)

        with torch.no_grad():
            next_actions, next_log_pi = self.actor.get_actions(batch_next_observations,
                                                               deterministic=False,
                                                               reparameterize=False)
            q1_next_target, q2_next_target = self.critic_target(batch_next_observations, 
                                                                next_actions)
            q_next_target = torch.min(q1_next_target, q2_next_target)
            alpha         = self.log_alpha.exp()
            q_target      = batch_rewards + (1 - batch_dones) * self.config.gamma * (q_next_target - alpha * next_log_pi)

        # Bellman loss
        q1_pred, q2_pred = self.critic(batch_observations, batch_actions.float())
        bellman_loss     = F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)
        return bellman_loss
    
    def compute_actor_and_alpha_loss(self, batch):
        batch_observations = torch.tensor(batch[0], dtype=torch.float32).to(self._device)
        policy_actions, curr_log_pi = self.actor.get_actions(batch_observations,
                                                             deterministic=False,
                                                             reparameterize=True)
        alpha_loss = -(self.log_alpha * (curr_log_pi + self.target_entropy).detach()).mean()

        if self.config.optimize_alpha:
            self.opt_alpha.zero_grad()
            alpha_loss.backward()
            self.opt_alpha.step()

        alpha = self.log_alpha.exp()
        if self.logger is not None:
            self.logger.log({'alpha': alpha})

        q1, q2  = self.critic(batch_observations, 
                              policy_actions)
        Q_value = torch.min(q1, q2)
        actor_loss = (alpha * curr_log_pi - Q_value).mean()
        return actor_loss, alpha_loss

    def optimize_networks(self, batch):
        actor_loss, alpha_loss = self.compute_actor_and_alpha_loss(batch)
        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        critic_loss = self.compute_critic_loss(batch)
        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        if self.logger is not None:
            self.logger.log({'critic_loss': critic_loss.detach().cpu().float().item(),
                             'actor_loss':  actor_loss.detach().cpu().float().item(),
                             'alpha_loss':  alpha_loss.detach().cpu().float().item()})

    def train_step(self):
        batch = self.replay_buffer.sample(self.config.batch_size)

        # Update Actor and Critic
        self.optimize_networks(batch)

        # Soft Update
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.config.tau) + param.data * self.config.tau)
    
    def serialize(self):
        return {'actor': self.actor.state_dict(),
                'critic': self.critic.state_dict(),
                'critic_target': self.critic_target.state_dict(),
                'log_alpha': self.log_alpha.detach().cpu().float().item()}

    def deserialize(self, d : dict):
        self.actor.load_state_dict(d['actor'])
        self.critic.load_state_dict(d['critic'])
        self.critic_target.load_state_dict(d['critic_target'])
        self.log_alpha = nn.Parameter(torch.Tensor([d['log_alpha']]), 
                                      requires_grad=True)

    def save(self, path):
        torch.save(self.serialize(), path)

    def load(self, path):
        """Bad architecture. Load should be @classmethod and perform full instantiation."""
        self.deserialize(torch.load(path))
