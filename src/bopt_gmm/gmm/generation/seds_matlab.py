try:
    import matlab.engine

    HAS_MATLAB = True
except ModuleNotFoundError as e:
    print(f'MATLAB not found: {e}')
    HAS_MATLAB = False

import numpy as np

from pathlib import Path

from bopt_gmm.gmm import GMM


class SEDS_MATLAB(object):
    def __init__(self, seds_path):
        if not HAS_MATLAB:
            raise Exception(f'Cannot instantiate SEDS Matlab, due to MATLAB being missing')
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(str(Path(__file__).absolute().parent))
        self.seds_path = seds_path

    def __del__(self):
        self.eng.quit()

    def fit_model(self, x0, xT, data, idcs, n_priors=3, objective='likelihood', dt=1/100, tol_cutting=0.1, max_iter=600, gmm_type=GMM, model_kwargs={}):
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
        return gmm_type(priors, mu.T, sigma.T, **model_kwargs)

def gen_seds_data_from_transitions(transition_trajectories, deltaT, use_force=False):
    data    = []
    x_zeros = []
    x_ts    = []
    o_idx   = []
    for t in transition_trajectories:
        out_traj = []
        for x, (pobs, _, _, _, _) in enumerate(t):
            if x == 0: # Skip because we cannot generate vel
                if use_force and 'force' in pobs:
                    x_zeros.append(np.hstack((pobs['position'], pobs['force'])))
                continue
            p = pobs['position'] if not use_force or not 'force' in pobs else np.hstack((pobs['position'], pobs['force']))
            p_t1 = t[x-1][0]['position'] if not use_force or not 'force' in t[x-1][0] else np.hstack((t[x-1][0]['position'], t[x-1][0]['force']))
            v = (p - p_t1) / deltaT
            # out_traj.append(np.hstack((p, v, pobs['force'])))
            out_traj.append(np.hstack((p, v)))
        x_ts.append(p)
        o_idx.append(sum([len(d) for d in data]))
        data.append(np.vstack(out_traj)[10:])
        # data[-1][:,-2] = gauss_smoothing(data[-1][:,-2], 4)
        # data[-1][:,-1] = gauss_smoothing(data[-1][:,-1], 4)
        # plt.plot(data[-1][:,-2], color='green', label='Force 1')
    #     plt.plot(data[-1][:,-1], color='blue', label='Force 2')
    # plt.show()
    o_idx.append(sum([len(d) for d in data]))
    print(f'Means: {np.mean(x_ts, axis=0)}\nStd: {np.std(x_ts, axis=0)}')

    return np.vstack(x_zeros), np.vstack(x_ts), np.vstack(data), np.array(o_idx)

def gen_seds_data_from_trajectories(trajs, delta_t):
    data    = []
    x_zeros = []
    x_ts    = []
    o_idx   = []
    for t in trajs:
        x_zeros.append(t[0])
        x_ts.append(t[-1])
        o_idx.append(sum([len(d) for d in data]))
        pv_traj = np.hstack((t[1:], (t[1:] - t[:-1]) / delta_t))
        data.append(pv_traj)

    return np.vstack(x_zeros), np.vstack(x_ts), np.vstack(data), np.array(o_idx)

def gen_trajectories_from_npz(f_trajs, delta_t):
    return gen_seds_data_from_trajectories([t['position'] for t in f_trajs], delta_t)

def seds_data_prep_last_step(x_zeros, x_ts, data, o_idx):
    return np.mean(x_zeros, axis=0), np.mean(x_ts, axis=0), data, o_idx

def print_seds_data_analysis(x_zeros, x_ts, points, indices):
    print( 'Starting points:\n'
          f'  Mean: {np.mean(x_zeros, axis=0)}\n'
          f'    SD: {np.std(x_zeros, axis=0)}\n'
          f'   Min: {np.min(x_zeros, axis=0)}\n'
          f'   Max: {np.max(x_zeros, axis=0)}\n'
           'Final points:\n'
          f'  Mean: {np.mean(x_ts, axis=0)}\n'
          f'    SD: {np.std(x_ts, axis=0)}\n'
          f'   Min: {np.min(x_ts, axis=0)}\n'
          f'   Max: {np.max(x_ts, axis=0)}\n'
           'Data:\n'
          f'  Mean: {np.mean(points, axis=0)}\n'
          f'    SD: {np.std(points, axis=0)}\n'
          f'   Min: {np.min(points, axis=0)}\n'
          f'   Max: {np.max(points, axis=0)}\n')

# ------ SINGELTON SEDS GENERATOR -------
SEDS = None

def seds_gmm_generator(seds_path, gmm_type, n_priors, objective='likelihood', tol_cutting=0.1, max_iter=600):
    # Ugly, I know
    global SEDS
    
    if SEDS is None:
        SEDS = SEDS_MATLAB(seds_path)

    def seds_generator(transitions, delta_t):
        x0s, xTs, data, oIdx = gen_seds_data_from_transitions(transitions, delta_t, True)
        x0, xT, data, oIdx   = seds_data_prep_last_step(x0s, xTs, data, oIdx)
        gmm = SEDS.fit_model(x0, xT, data, oIdx, n_priors, 
                             objective=objective, dt=delta_t, 
                             tol_cutting=tol_cutting, max_iter=max_iter, gmm_type=gmm_type)
        return gmm
    
    return seds_generator