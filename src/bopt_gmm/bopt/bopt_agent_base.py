import numpy as np

from dataclasses import dataclass
from datetime    import datetime
from skopt       import Optimizer
from typing      import Callable, Any, Iterable, Tuple

import bopt_gmm.gmm as lib_gmm
from bopt_gmm.logging import LoggerBase
from drlfads.lpvDS.seds      import SEDS_MATLAB, \
                                    gen_seds_data_from_trajectories, \
                                    gen_seds_data_from_transitions,  \
                                    seds_data_prep_last_step


def base_success(prior_obs, posterior_obs, action, reward, done):
    return done and reward > 0

def base_gen_gmm(trajectories, delta_t):
    raise NotImplementedError

def no_op(_, x):
    return x


BOPT_TIME_SCALE = 'bopt_step'

@dataclass
class BOPTAgentConfig:
    f_success   : Callable[[Any, Any, Any, float, bool], bool] = base_success
    prior_range : float = 0.15
    mean_range  : float = 0.05
    early_tell  : int   = 5
    late_tell   : int   = 100000


class BOPTGMMAgentBase(object):
    @dataclass
    class BOPTState:
        step    : int   = 0
        updates : int   = 0
        reward  : float = 0.0
    
    @dataclass
    class State:
        n_step : int         = 0
        trajectories         = [[]]
        success_trajectories = []
        gp_optimizer         = None
        current_update       = None
        bopt_state           = None
        obs_transform        = no_op


    def __init__(self, base_gmm, config: BOPTAgentConfig, logger : LoggerBase=None) -> None:
        self.base_model = base_gmm
        self.config     = config
        self.model      = base_gmm
        self.logger     = logger

        self.state = BOPTGMMAgentBase.State()

        if self.logger is not None:
            self.logger.define_metric('bopt_x', BOPT_TIME_SCALE)
            self.logger.define_metric('bopt_y', BOPT_TIME_SCALE)

    def predict(self, observation):
        observation = self.state.obs_transform(observation)
        return {'motion': self.model.predict(observation).flatten(), 'gripper': 0.1}

    def step(self, prior_obs, posterior_obs, action, reward, done):
        if reward != 0:
            print(f'Reward! {reward}')
        transition = (prior_obs, posterior_obs, action, reward, done)
        self.state.n_step += 1
        self.state.trajectories[-1].append(transition)
        if self.config.f_success(*transition):
            self.state.success_trajectories.append(self.state.trajectories[-1])
            print(f'Collected a a successful trajectory. Now got {len(self.state.success_trajectories)}')
        if done:
            self.state.trajectories.append([])
            
            # If GP-Optimization is in progress, step it!
            if self.state.current_update is not None:
                self.step_optimizer(reward)

    def init_optimizer(self, base_accuracy=None):
        self.state.bopt_state    = BOPTGMMAgentBase.BOPTState()
        self.state.base_accuracy = base_accuracy if base_accuracy is not None else len(self.state.success_trajectories) / len(self.state.trajectories)

        self.state.gp_optimizer  = Optimizer([(-self.config.prior_range, self.config.prior_range)] * self.base_model.n_priors + 
                                             [(-self.config.mean_range, self.config.mean_range)] * self.base_model.n_priors * self.base_model.n_dims)
        
        print(f'Base Model:\nPriors: {self.base_model.pi()}\nMu: {self.base_model.mu()}')

        self.update_model()

    def step_optimizer(self, reward):
        self.state.bopt_state.step   += 1
        self.state.bopt_state.reward += reward

        if self.state.bopt_state.step % min(max(int(1.0 / self.state.base_accuracy), self.config.early_tell), self.config.late_tell) == 0:
            print(f"Finished gp run {self.state.bopt_state.updates} ({self.state.bopt_state.step}). Return: {self.state.bopt_state.reward}. Let's go again!")
            self._tell(self.state.current_update, self.state.bopt_state.reward)    
            self.update_model()

    def update_model(self):
        for x in range(100):
            try:
                self.state.current_update = self.state.gp_optimizer.ask()

                mu = self.base_model.mu()
                mu_space = mu.max(axis=0) - mu.min(axis=0)

                self.model = self.base_model.update_gaussian(self.state.current_update[:self.base_model.n_priors], 
                                                            np.asarray(self.state.current_update[self.base_model.n_priors:]).reshape(mu.shape) * mu_space)
                
                print(f'Updated Model:\nPriors: {self.model.pi()}\nMu: {self.model.mu()}')
                break
            except ValueError:
                self.state.gp_optimizer.tell(self.state.current_update, 0)
                self.state.bopt_state.updates += 1
        else:
            raise Exception(f'Repeated Bayesian Updates have failed to produce a valid update')

    def _tell(self, state, reward):
        self.state.gp_optimizer.tell(state, -reward)
        self.state.bopt_state.updates += 1
        self.state.bopt_state.reward   = 0.0
        if self.logger is not None:
            self.logger.log({
                BOPT_TIME_SCALE: self.state.bopt_state.updates,
                'bopt_y': self.state.gp_optimizer.yi[-1],
                'bopt_x': self.state.gp_optimizer.Xi[-1]
            })

    def reset(self):
        self.state = type(self.state)()
        self.model = self.base_model

    def has_gp_stage(self):
        return True

    def is_in_gp_stage(self):
        return self.state.gp_optimizer is not None

    def get_bopt_step(self):
        return self.state.bopt_state.updates if self.state.bopt_state is not None else 0


def gen_gmm_from_trajectories(trajectories, deltaT, n_priors, gmm_type):
    data = np.vstack(gen_trajectory_from_transitions(trajectories, deltaT))
    return gmm_fit_em(n_priors, data, gmm_type)

