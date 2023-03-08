import numpy as np

from collections import OrderedDict

from functools import lru_cache
from .gmm import GMM, add_gmm_model

class GMMCart3D(GMM):
    GIVEN = (0, 1, 2)

    def predict(self, obs_dict, dims=GIVEN, full=False):
        if type(obs_dict) == dict:
            obs_dict = obs_dict['position']
        return super().predict(obs_dict, dims)

    def get_weights(self, obs, dims=None):
        if type(obs) == dict:
            obs = obs['position']
        return super().get_weights(obs, dims)

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

    @lru_cache(1)
    def semantic_dims(self):
        return OrderedDict([('position', self.state_dim), 
                            ('velocity', self.prediction_dim)])

    @lru_cache(1)
    def semantic_obs_dims(self):
        return OrderedDict([('position', self.state_dim[:3])])

add_gmm_model(GMMCart3D)

class GMMCart3DJS(GMM):
    def __init__(self, priors, means, cvar, dim_names=[]):
        super().__init__(priors, means, cvar)
        self._dim_names = dim_names
        self.GIVEN = tuple(range(3 + len(dim_names)))

    def predict(self, obs_dict, dims=None, full=False):
        dims = self.GIVEN if dims is None else dims
        if type(obs_dict) == dict:
            obs_dict = np.hstack([obs_dict[x] for x in ['position'] + self._dim_names])

        return super().predict(obs_dict, dims)[:,:3] if not full else super().predict(obs_dict, dims)

    def get_weights(self, obs, dims=None):
        if type(obs) == dict:
            obs = np.hstack([obs[x] for x in ['position'] + self._dim_names])
        return super().get_weights(obs, dims)

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

    def _custom_data(self):
        d = super()._custom_data()
        d.update({'dim_names': self._dim_names})
        return d

    @property
    def state_dim(self):
        return self.GIVEN

    @property
    @lru_cache(1)
    def prediction_dim(self):
        return tuple([x + len(self.state_dim) for x in range(len(self.state_dim))])

    @lru_cache(1)
    def semantic_dims(self):
        out = OrderedDict([('position', self.state_dim),
                           ('velocity', self.prediction_dim)])
        for x, d in enumerate(self._dim_names):
            out[d] = (x + len(self.state_dim) - len(self._dim_names),)
        
        for x, d in enumerate(self._dim_names):
            out[f'{d}_vel'] = (x + 2 * len(self.state_dim) - len(self._dim_names),)

        return out

    @lru_cache(1)
    def semantic_obs_dims(self):
        out = OrderedDict([('position', self.state_dim[:3])])
        for x, d in enumerate(self._dim_names):
            out[d] = (x + len(self.state_dim) - len(self._dim_names),)
        return out

add_gmm_model(GMMCart3DJS)


class GMMCart3DForce(GMM):
    GIVEN = (0, 1, 2, 3, 4, 5)

    def __init__(self, priors, means=12, cvar=None, force_scale=1.0):
        super().__init__(priors, means, cvar)
        self._force_scale = force_scale

    def predict(self, obs_dict, dims=GIVEN, full=False):
        if type(obs_dict) == dict:
            obs_dict = np.hstack((obs_dict['position'], obs_dict['force']))
        scale = np.isin(dims, self.GIVEN[-3:], invert=True) + np.isin(dims, self.GIVEN[-3:]).astype(float) * self._force_scale
        return super().predict(obs_dict * scale, dims)[:,:3] if not full else super().predict(obs_dict * scale, dims)

    def get_weights(self, obs, dims=None):
        if type(obs) == dict:
            obs = np.hstack((obs['position'], obs['force']))
        scale = np.isin(dims, self.GIVEN[-3:], invert=True) + np.isin(dims, self.GIVEN[-3:]).astype(float) * self._force_scale
        return super().get_weights(obs * scale, dims)

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
        return (6, 7, 8, 9, 10, 11)

    def conditional_pdf(self, X: np.array, d_given, *Ys: np.array):
        d_predicted = [x for x in set(range(len(self.GIVEN) * 2)) if x not in d_given]
        
        scale_x = np.isin(d_given, self.GIVEN[-3:] + self.prediction_dim[-3:], invert=True) + np.isin(d_given, self.GIVEN[-3:] + self.prediction_dim[-3:]).astype(float) * self._force_scale
        scale_y = np.isin(d_predicted, self.GIVEN[-3:] + self.prediction_dim[-3:], invert=True) + np.isin(d_predicted, self.GIVEN[-3:] + self.prediction_dim[-3:]).astype(float) * self._force_scale
        
        return super().conditional_pdf(X * scale_x, d_given, *[y * scale_y for y in Ys])

    def pdf(self, points : np.ndarray, dims=None):
        scale  = np.isin(dims, self.GIVEN[-3:] + self.prediction_dim[-3:], invert=True) + np.isin(dims, self.GIVEN[-3:] + self.prediction_dim[-3:]).astype(float) * self._force_scale
        return super().pdf(points * scale, dims)

    def _custom_data(self):
        d = super()._custom_data()
        d.update({'force_scale': self._force_scale})
        return d

    def update_gaussian(self, priors=None, mu=None, sigma=None, sigma_scale=None, sigma_eigen_update=None, sigma_rotation=None):
        new_model = super().update_gaussian(priors, mu, sigma, sigma_scale, sigma_eigen_update, sigma_rotation=sigma_rotation)
        new_model._force_scale = self._force_scale
        return new_model

    @lru_cache(1)
    def semantic_dims(self):
        return OrderedDict([('position', self.state_dim[:3]),
                            ('force', self.state_dim[3:]), 
                            ('velocity', self.prediction_dim[:3]),
                            ('force_vel', self.prediction_dim[3:])])
    
    @lru_cache(1)
    def semantic_obs_dims(self):
        return OrderedDict([('position', self.state_dim[:3]),
                            ('force', self.state_dim[3:])])

add_gmm_model(GMMCart3DForce)


class GMMCart3DTorque(GMMCart3DForce):
    def predict(self, obs_dict, dims=GMMCart3DForce.GIVEN, full=False):
        if type(obs_dict) == dict:
            obs_dict = np.hstack((obs_dict['position'], obs_dict['torque']))
        scale = np.isin(dims, self.GIVEN[-3:], invert=True) + np.isin(dims, self.GIVEN[-3:]).astype(float) * self._force_scale
        return super().predict(obs_dict * scale, dims)[:,:3] if not full else super().predict(obs_dict * scale, dims)

    def get_weights(self, obs, dims=None):
        if type(obs) == dict:
            obs = np.hstack((obs['position'], obs['torque']))
        scale = np.isin(dims, self.GIVEN[-3:], invert=True) + np.isin(dims, self.GIVEN[-3:]).astype(float) * self._force_scale
        return super().get_weights(obs * scale, dims)

    @lru_cache(1)
    def semantic_dims(self):
        return OrderedDict([('position', self.state_dim[:3]),
                            ('torque', self.state_dim[3:]), 
                            ('velocity', self.prediction_dim[:3]),
                            ('torque_vel', self.prediction_dim[3:])])

add_gmm_model(GMMCart3DTorque)


def load_gmm(gmm_config):
    return GMM.load_model(gmm_config.model)
