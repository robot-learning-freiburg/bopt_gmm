import numpy as np

from scipy.stats import multivariate_normal
from sklearn.mixture import BayesianGaussianMixture

from bopt_gmm.gmm   import GMM
from bopt_gmm.utils import unpack_transition_traj, \
                           calculate_trajectory_velocities

def gmm_fit_em(n_components, points, gmm_type=GMM, max_iter=100, tol=0.01, n_init=1):
    """EM-Algorithm for fitting a GMM to data

    Args:
        n_components (int): Number of components to use for GMM
        points (np.ndarray): Data to fit to. (points, dim)
    """

    bgmm = BayesianGaussianMixture(n_components=n_components,
                                   max_iter=max_iter,
                                   random_state=np.random.RandomState(0),
                                   tol=tol,
                                   n_init=n_init).fit(points)

    # means = points[np.random.choice(points.shape[0], n_components)]
    # gmm   = GMM(n_components, means)

    # # probability of points belonging to components (priors, points)
    # probs        = gmm.pdf(points)
    # prior_mass   = probs.sum(axis=1)
    # # New prior weights
    # prior_weight = prior_mass / prior_mass.sum()
    # # Points scaled per prior (priors, dims, points)
    # weighted_points = np.expand_dims(probs, axis=-1) * np.expand_dims(points.T, axis=0)
    # new_means       = (1 / prior_weight) * weighted_points.sum(axis=2)

    # # Deltas (priors, dims, points)
    # new_delta = np.expand_dims(new_means, -1) - np.expand_dims(points.T, 0)

    # new_cvars = np.zeros(gmm._cvar.shape)

    # for x, delta in enumerate(new_delta):
    #     sigmas       = np.stack([d.dot(d.T) for d in np.expand_dims(delta.T, -1)]) * probs[x]
    #     new_cvars[x] = sigmas.sum(axis=0)

    # new_gmm = GMM(prior_weight, new_means, new_cvars)

    # cov = nearestPD(bgmm.covariances_) if not isPD(bgmm.covariances_) else bgmm.covariances_

    return gmm_type(bgmm.weights_, bgmm.means_, bgmm.covariances_)

def em_gmm_generator(gmm_type, n_priors, max_iter, data_order=['position', 'force', 'torque']):
    def em_generator(transitions, delta_t):
        trajs = [unpack_transition_traj(t) for t in transitions]

        fields, groups, _ = trajs[0]
        group_prefixes = [fields[g[0]][:fields[g[0]].rfind('_')] for g in groups]

        if data_order is None:
            group_order = list(range(len(group_prefixes)))
        else:
            data_order  = [d for d in data_order if d in group_prefixes]
            group_order = [group_prefixes.index(d) for d in data_order]
        
        data_points = []

        for _, groups, _, data in trajs:
            ordered_data = np.hstack([np.take(data, groups[go], axis=1) for go in group_order])
            data_points.append(calculate_trajectory_velocities(ordered_data, delta_t))

        data_points = np.vstack(data_points)
        return gmm_fit_em(n_priors, data_points, gmm_type)
    
    return em_generator