import numpy as np

class BoxSampler(object):
    def __init__(self, b_min, b_max):
        if len(b_min) != len(b_max):
            raise Exception(f'Box bounds need to be the same size. Min: {len(b_min)} Max: {len(b_max)}')

        self.b_min = b_min
        self.b_max = b_max
    
    def sample(self):
        return (np.random.random(len(self.b_min)) * (np.asarray(self.b_max) - np.asarray(self.b_min))) + np.asarray(self.b_min)


class NoiseSampler(object):
    def __init__(self, dim, var, constant) -> None:
        self.constant = constant
        self.dim = dim
        self.var = var
        self._noise = None
        self.reset()
    
    def sample(self):
        if self.constant and self._noise is not None:
            return self._noise
        return np.random.normal(0, self.var, self.dim)
        
    def reset(self):
        self._noise = None
        self._noise = self.sample()
