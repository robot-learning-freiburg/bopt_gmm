from bopt_gmm.bopt import GMMOptAgent

from stable_baselines3.sac import MlpPolicy


class SACGMMAgent(MlpPolicy):
    def __init__(self, gmm, config, obs_space, gripper_command=0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self._gmm_agent = GMMOptAgent(gmm, config)
        self.pseudo_bopt_step = 0
        self._gripper_command = gripper_command
        self._sac = MlpPolicy()

    def forward(self, obs, deterministic=False):
        pass

    def predict(self, obs):
        # if callable(self._force_norm):
        #     obs = self._force_norm(obs)
        # elif 'force' in obs:
        #     obs['force'] = obs['force'] * self._force_norm
        return {'motion': self._model.predict(obs).flatten(), 'gripper': self._gripper_command}

    def step(self, *args):
        pass
    
    def has_gp_stage(self):
        return False

    def is_in_gp_stage(self):
        return False

    def get_bopt_step(self):
        self.pseudo_bopt_step += 1
        return self.pseudo_bopt_step
