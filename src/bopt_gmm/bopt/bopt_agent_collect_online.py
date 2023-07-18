import numpy as np

from dataclasses import dataclass
from datetime    import datetime
from typing      import Callable, Any, Iterable, Tuple

import bopt_gmm.gmm as lib_gmm

from .bopt_agent_base import BOPTGMMAgentBase, \
                             BOPTAgentConfig

from bopt_gmm.utils import unpack_trajectories, \
                           calculate_trajectory_velocities, \
                           normalize_trajectories


def base_gen_gmm(trajectories, delta_t):
    raise NotImplementedError

@dataclass
class BOPTAgentGenGMMConfig(BOPTAgentConfig):
    n_successes     : int = 10
    f_gen_gmm       : Callable[[Iterable[Tuple[Any, Any, Any, float, bool]], float], lib_gmm.GMM] = base_gen_gmm
    delta_t         : float = 0.05
    debug_data_path : str = None
    debug_gmm_path  : str = None


class BOPTGMMCollectAndOptAgent(BOPTGMMAgentBase):
    
    def step(self, prior_obs, posterior_obs, action, reward, done):
        super().step(prior_obs, posterior_obs, action, reward, done)

        # Start optimization process
        if len(self.state.success_trajectories) == self.config.n_successes and not self.is_in_gp_stage():
            print('Starting Bayesian optimization')
        
            stamp = datetime.now()

            if self.config.debug_data_path is not None:
                np.savez(f'{self.config.debug_data_path}_{stamp}.npz', self.state.success_trajectories)
                print(f'Saved success trajectories to "{self.config.debug_data_path}"')

            self.base_model = self.config.f_gen_gmm(self.state.success_trajectories, self.config.delta_t)
            
            if self.config.debug_gmm_path is not None:
                self.base_model.save_model(f'{self.config.debug_gmm_path}_{stamp}.npy')

            self.init_optimizer()
            

    # @staticmethod
    # def calculate_force_normalization(trajectories):
    #     pos_force = np.abs(np.vstack([np.vstack([np.hstack((p['position'], p['force'])) for p, _, _, _, _ in t]) for t in trajectories])).max(axis=0)
        
    #     return (pos_force[:3] / pos_force[3:]).max()
    
    # @staticmethod
    # def normalize_force_trajectories(factor, trajectories):
    #     out = []
    #     for t in trajectories:
    #         nt = []
    #         for prior, posterior, action, reward, done in t:
    #             prior = {'position': prior['position'],
    #                         'force': prior['force'] * factor}
    #             posterior = {'position': posterior['position'],
    #                             'force': posterior['force'] * factor}
    #             nt.append((prior, posterior, action, reward, done))
    #         out.append(nt)
    #     return out

