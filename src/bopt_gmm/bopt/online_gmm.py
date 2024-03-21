import numpy as np

from dataclasses import dataclass
from datetime    import datetime
from typing      import Callable, Any, Iterable, Tuple
from bopt_gmm.bopt.bopt_agent_base import BOPTAgentConfig, no_op

import bopt_gmm.gmm as lib_gmm
from bopt_gmm.logging import LoggerBase

from .bopt_agent_base import BOPTGMMAgentBase, \
                             base_success,     \
                             BOPT_TIME_SCALE

from bopt_gmm.utils import unpack_trajectories, \
                           calculate_trajectory_velocities, \
                           normalize_trajectories


def base_gen_gmm(trajectories, delta_t):
    raise NotImplementedError

@dataclass
class OnlineGMMConfig:
    original_data   : Iterable[Tuple[Any, Any, Any, float, bool]]
    f_success       : Callable[[Any, Any, Any, float, bool], bool] = base_success
    f_gen_gmm       : Callable[[Iterable[Tuple[Any, Any, Any, float, bool]], float], lib_gmm.GMM] = base_gen_gmm
    gripper_command : float = 0.5
    delta_t         : float = 0.05
    debug_gmm_path  : str   = None
    prior_range     : float = 0.0
    mean_range      : float = 0.0
    sigma_range     : float = 0.0

class OnlineGMMAgent(BOPTGMMAgentBase):
    def __init__(self, base_gmm, config: BOPTAgentConfig, obs_transform=no_op, logger: LoggerBase = None) -> None:
        super().__init__(base_gmm, config, obs_transform, logger)

        self._pseudo_config = None

    def step(self, prior_obs, posterior_obs, action, reward, done):
        # Hacking our way past init_optimizer
        if self.state.bopt_state is None:
            self.state.bopt_state = BOPTGMMAgentBase.BOPTState()
        # Super lazy check for new data
        n_successes = len(self.state.success_trajectories)

        super().step(prior_obs, posterior_obs, action, reward, done)

        # New data
        if n_successes < len(self.state.success_trajectories):
            print('Retraining model...')
            self.model = self.config.f_gen_gmm(self.config.original_data + self.state.success_trajectories,
                                               self.config.delta_t, 
                                               self.base_model)
            print('Done!')
            # Increment bopt update for logging
            self.state.bopt_state.updates += 1
            self._pseudo_config = {'n_trajectories': self.state.bopt_state.updates}

            if self.logger is not None:
                self.logger.log({
                    BOPT_TIME_SCALE: self.state.bopt_state.updates,
                    'bopt_y': reward,
            #         'bopt_x': self.state.gp_optimizer.Xi[-1]
                })

        # if self.config.debug_gmm_path is not None:
        #     self.base_model.save_model(f'{self.config.debug_gmm_path}_{stamp}.npy')
    
    def get_incumbent(self):
        return self.model

    def get_incumbent_config(self):
        return self._pseudo_config

