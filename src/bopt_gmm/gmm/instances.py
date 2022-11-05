import numpy as np

from .gmm import GMM

class GMMCart3D(GMM):
    GIVEN = [0, 1, 2]

    def predict(self, obs_dict):
        return super().predict(obs_dict['position'], self.GIVEN)

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
        return [3, 4, 5]


class GMMCart3DForce(GMM):
    GIVEN = [0, 1, 2, 3, 4, 5]

    def predict(self, obs_dict):
        return super().predict(np.hstack((obs_dict['position'], obs_dict['force'])), self.GIVEN)[:,:3]

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
        return [6, 7, 8, 9, 10, 11]


class GMMCart3DTorque(GMMCart3DForce):
    def predict(self, obs_dict):
        return super().super().predict(np.hstack((obs_dict['position'], obs_dict['torque'])), self.GIVEN)[:,:3]
