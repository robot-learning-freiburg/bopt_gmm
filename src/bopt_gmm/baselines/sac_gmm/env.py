import numpy as np

from bopt_gmm.bopt import GMMOptAgent
from dataclasses   import dataclass
from functools     import lru_cache
from gym           import Env
from gym.spaces    import Box
from typing        import List

from bopt_gmm.bopt import BOPT_TIME_SCALE


def f_void(*args):
    pass


@dataclass
class SACGMMEnvCallback:
    on_episode_start = f_void
    on_episode_end   = f_void
    on_reset         = f_void
    on_post_step     = f_void



class SACGMMEnv(Env):
    def __init__(self, env : Env, 
                 gmm_agent : GMMOptAgent, 
                 gripper_command, 
                 sacgmm_config,
                 obs_filter={'position', 'force'}, 
                 cb : List[SACGMMEnvCallback] = None):
        self.agent  = gmm_agent
        self.env    = env
        self.config = sacgmm_config
        self._n_env_steps = 0
        self._ep_count  = 0
        self._sac_steps = 0
        self._action_dims = None
        self._obs_dims    = None
        self._obs_filter  = obs_filter
        self._gripper_command = gripper_command
        self._cb = cb if cb is not None else []
        
        # Generate internals
        self.observation_space
        self.action_space

    def add_callback(self, cb):
        self._cb.append(cb)
    
    def remove_callback(self, cb):
        self._cb.remove(cb)

    @property
    def config_space(self):
        return self.env.config_space

    def config_dict(self):
        return self.env.config_dict()

    @property
    @lru_cache(1)
    def observation_space(self):
        lb = []
        ub = []
        self._obs_dims = []

        for k, bound in self.env.observation_space.items():
            if k in self._obs_filter:
                self._obs_dims.append(k)
                lb.append(bound.low)
                ub.append(bound.high)

        return Box(low=np.hstack(lb),
                   high=np.hstack(ub))

    @property
    @lru_cache(1)
    def action_space(self):
        lb = []
        ub = []
        self._action_dims = []

        for k, bound in self.agent.config_space.items():
            self._action_dims.append(k)
            lb.append(bound.lower)
            ub.append(bound.upper)

        return Box(low=np.asarray(lb),
                   high=np.asarray(ub))

    def _convert_action(self, action_dict):
        return np.hstack([action_dict[k] for k in self._action_dims])

    def _convert_obs(self, obs_dict):
        return np.hstack([obs_dict[d] for d in self._obs_dims])

    def reset(self):
        self.agent.reset()
        obs = self.env.reset()
        self._n_env_steps = 0
        obs = self._convert_obs(obs)
        
        if self._cb is not None:
            for cb in self._cb:
                cb.on_reset(obs)
        return obs

    def step(self, action):
        dict_update = dict(zip(self._action_dims, action))

        if self._n_env_steps == 0:
            for cb in self._cb:
                cb.on_episode_start(self.env.config_dict, dict_update)

        self.agent.update_model(dict_update)

        obs = self.env.observation()

        for x in range(self.config.sacgmm_steps):
            action = self.agent.predict(obs)
            action = {'motion': action.flatten(), 
                      'gripper': self._gripper_command}
            # print(observation)
            obs, reward, done, info = self.env.step(action)
            
            if done:
                break
        
        self._n_env_steps += x + 1
        self._sac_steps   += 1
        
        for cb in self._cb:
            cb.on_post_step(reward, self._n_env_steps)

        done = done or self._n_env_steps >= self.config.episode_steps
        
        if done:
            self._ep_count += 1

            for cb in self._cb:
                cb.on_episode_end(self.env, obs, self._n_env_steps, self._sac_steps)        

        obs = self._convert_obs(obs)

        return obs, reward, done, info

    def eval_copy(self):
        out = SACGMMEnv(self.env,  
                        self.agent, 
                        self._gripper_command, 
                        self.config, 
                        self._obs_filter)
        # Generate internal variables
        os  = out.observation_space
        acs = out.action_space
        return out
