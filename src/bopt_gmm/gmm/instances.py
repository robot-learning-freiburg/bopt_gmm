import numpy as np

from .gmm import GMM, add_gmm_model

class GMMCart3D(GMM):
    GIVEN = (0, 1, 2)

    def predict(self, obs_dict, dims=GIVEN, full=False):
        if type(obs_dict) == dict:
            obs_dict = obs_dict['position']
        return super().predict(obs_dict, dims)

    def mu_pos(self):
        return self.mu([0, 1, 2])

    def mu_vel(self):
        return self.mu([3, 4, 5])

    def sigma_pos(self):
        return self.sigma([0, 1, 2], [0, 1, 2])
    
    def sigma_vel(self):
        return self.sigma([3, 4, 5], [3, 4, 5])
    
    def sigma_pos_vel(self):
        return self.sigma([0, 1, 2], [3, 4, 5])
    
    def sigma_vel_pos(self):
        return self.sigma([3, 4, 5], [0, 1, 2])

    @property
    def state_dim(self):
        return self.GIVEN

    @property
    def prediction_dim(self):
        return (3, 4, 5)

add_gmm_model(GMMCart3D)


class GMMCart3DForce(GMM):
    GIVEN = (0, 1, 2, 3, 4, 5)

    def __init__(self, priors, means, cvar, force_scale=1.0):
        super().__init__(priors, means, cvar)
        self._force_scale = force_scale

    def predict(self, obs_dict, dims=GIVEN, full=False):
        if type(obs_dict) == dict:
            obs_dict = np.hstack((obs_dict['position'], obs_dict['force']))
        obs_dict[3:6] *= self._force_scale
        return super().predict(obs_dict, dims)[:,:3] if not full else super().predict(obs_dict, dims)

    def mu_pos(self):
        return self.mu([0, 1, 2])

    def mu_vel(self):
        return self.mu([6, 7, 8])

    def mu_f(self):
        return self.mu([3, 4, 5])

    def sigma_pos(self):
        return self.sigma([0, 1, 2], [0, 1, 2])
    
    def sigma_vel(self):
        return self.sigma([6, 7, 8], [6, 7, 8])
    
    def sigma_f(self):
        return self.sigma([3, 4, 5], [3, 4, 5])
    
    def sigma_posf_vel(self):
        return self.sigma([0, 1, 2, 3, 4, 5], [6, 7, 8])
    
    def sigma_vel_pos(self):
        return self.sigma([6, 7, 8], [0, 1, 2])

    @property
    def state_dim(self):
        return self.GIVEN

    @property
    def prediction_dim(self):
        return (6, 7, 8) #, 9, 10, 11]

    def _custom_data(self):
        d = super()._custom_data()
        d.update({'force_scale': self._force_scale})
        return d

add_gmm_model(GMMCart3DForce)


class GMMCart3DTorque(GMMCart3DForce):
    def predict(self, obs_dict, dims=GMMCart3DForce.GIVEN, full=False):
        if type(obs_dict) == dict:
            obs_dict = np.hstack((obs_dict['position'], obs_dict['torque']))
        return super().predict(obs_dict, dims)[:,:3] if not full else super().predict(obs_dict, dims)

add_gmm_model(GMMCart3DTorque)


GMM_TYPES = {'position': GMMCart3D,
                'force': GMMCart3DForce,
               'torque': GMMCart3DTorque}


def load_gmm(gmm_config):
    return GMM_TYPES[gmm_config.type].load_model(gmm_config.model)
