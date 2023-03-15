import numpy as np

from scipy.stats import multivariate_normal

USE_SKLEARN = True

if USE_SKLEARN:
    from sklearn.mixture import BayesianGaussianMixture
else:
    import gmr

from bopt_gmm.gmm   import GMM
from bopt_gmm.utils import unpack_transition_traj, \
                           calculate_trajectory_velocities


def gmm_fit_em(n_components, points, gmm_type=GMM, max_iter=100, tol=0.01, n_init=1, prior_gmm=None, model_kwargs={}):
    """EM-Algorithm for fitting a GMM to data

    Args:
        n_components (int): Number of components to use for GMM
        points (np.ndarray): Data to fit to. (points, dim)
    """

    prior_weights = None if prior_gmm is None else prior_gmm.pi()
    prior_means   = None if prior_gmm is None else prior_gmm.mu()
    prior_cvars   = None if prior_gmm is None else prior_gmm.sigma()


    if USE_SKLEARN:
        bgmm = BayesianGaussianMixture(n_components=n_components,
                                       max_iter=max_iter,
                                       random_state=np.random.RandomState(0),
                                       tol=tol,
                                       n_init=n_init if prior_gmm is None else 1,
                                       reg_covar=1e-3)
        bgmm.fit(points)
        return gmm_type(bgmm.weights_, bgmm.means_, bgmm.covariances_, **model_kwargs)
    else:
        bgmm = gmr.GMM(n_components, prior_weights, prior_means, prior_cvars)

        bgmm.from_samples(points, n_iter=max_iter, init_params='kmeans++')
        return gmm_type(bgmm.priors, bgmm.means, bgmm.covariances, **model_kwargs)


def em_gmm_generator(gmm_type, n_priors, max_iter, tol, n_init, data_order=['position', 'force', 'torque']):
    def em_generator(transitions, delta_t):
        trajs = [unpack_transition_traj(t) for t in transitions]

        fields, groups, _ = trajs[0]
        group_prefixes = [fields[g[0]][:fields[g[0]].rfind('_')] for g in groups]

        ldo = data_order
        if ldo is None:
            group_order = list(range(len(group_prefixes)))
        else:
            ldo  = [d for d in ldo if d in group_prefixes]
            group_order = [group_prefixes.index(d) for d in ldo]
        
        data_points = []

        for _, groups, data in trajs:
            ordered_data = np.hstack([np.take(data, groups[go], axis=1) for go in group_order])
            data_points.append(calculate_trajectory_velocities(ordered_data, delta_t))

        data_points = np.vstack(data_points)
        return gmm_fit_em(n_priors, data_points, gmm_type, max_iter, tol, n_init)
    
    return em_generator