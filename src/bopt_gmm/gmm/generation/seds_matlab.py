
try:
    import matlab.engine
    import numpy as np

    HAS_MATLAB = True
except ModuleNotFoundError as e:
    print(f'MATLAB not found: {e}')
    HAS_MATLAB = False

class SEDS_MATLAB(object):
    def __init__(self, seds_path):
        if not HAS_MATLAB:
            raise Exception(f'Cannot instantiate SEDS Matlab, due to MATLAB being missing')
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(str(Path(__file__).absolute().parent))
        self.seds_path = seds_path

    def __del__(self):
        self.eng.quit()

    def fit_model(self, x0, xT, data, idcs, n_priors=3, objective='likelihood', dt=1/100, tol_cutting=0.1, max_iter=600, gmm_type=GMM):
        priors, mu, sigma = self.eng.gen_seds_model(self.seds_path, 
                                                    matlab.double(x0.T),
                                                    matlab.double(xT.T),
                                                    matlab.double(data.T),
                                                    matlab.int64(idcs),
                                                    n_priors,
                                                    objective,
                                                    dt,
                                                    tol_cutting,
                                                    max_iter,
                                                    nargout=3)
        priors = np.asarray(priors).squeeze()
        mu = np.asarray(mu)
        sigma = np.asarray(sigma)
        return gmm_type(priors, mu.T, sigma.T)
