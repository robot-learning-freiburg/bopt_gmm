import sys
import numpy as np

from argparse    import ArgumentError
from functools   import lru_cache
from itertools   import cycle
from pathlib     import Path
from random      import random
from scipy.stats import multivariate_normal
from typing      import Union

from .utils import isPD, nearestPD

class GMM(object):
    def __init__(self, priors: Union[int, np.array]=None, means: Union[int, np.array]=None, cvar: np.array=None):
        # 1D - Simply the priors
        n_components = priors if type(priors) == int else len(priors)
        if type(priors) == int:
            self._priors = np.ones(n_components) / n_components
        elif priors.shape != (n_components, ):
            raise Exception(f'Expected priors to be of shape ({n_components},) '
                            f'but they are of shape {priors.shape}')
        else:
            self._priors = priors / np.sum(priors)
        # 2D - Dimensionality * Number of components
        if means is None:
            raise ArgumentError(f'Means need to be either an integer indicating dimensionality, or specific means')
        
        if type(means) == int:
            self._means = np.zeros((n_components, means))
        else:
            self._means = means
        # 3D - Dimensionality^2 * Number of components
        dim_components = self._means.shape[1]

        if cvar is None:
            self._cvar   = np.stack([np.eye(dim_components) for _ in range(n_components)], 0)
        elif cvar.shape != (n_components, dim_components, dim_components):
            raise Exception(f'Expected covariances to be of shape {(n_components, dim_components, dim_components)} '
                            f'but they are of shape {cvar.shape}')
        else:
            self._cvar = cvar

    @property
    def n_priors(self):
        return len(self._priors)

    @property
    def n_dims(self):
        return self._means.shape[1]

    def get_weights(self, x: np.array, dims=None):
        """Calculates the weights of the mixture components for a given point x.

        Args:
            x (np.array): Point to calculate weights for.
            dims (iterable, optional): Dimensions to use for inference.  If point x is only 
                                       partial wrt the full state space of the GMM, this
                                       parameter holds the dimensions wrt the values in x are
                                       given. Default is to assume x to be fully defined.

        Raises:
            Exception: Raised when lower dimensional point is given without any dim-hint.

        Returns:
            np.array: Weights of the individual GMM components as they pertain to x.
        """
        if dims is None:
            if x.shape[1] < self._means.shape[1]: # Todo: Check fails for 1d inputs
                raise Exception(f'Dimensions for weighting need to '
                                f'be specified explicitly when given'
                                f'data is of lower dimensionality.')
                
            dims = range(self._means.shape[1])
        
        x = x if x.ndim == 2 else x.reshape((1, x.size))

        weights = np.zeros((self._priors.size, x.shape[0]))
        # Filtered covariances
        f_cvar  = self._cvar.take(dims, axis=1).take(dims, axis=2)
        for k in range(self._priors.size):
            c_mean = self._means[k].take(dims)
            c_cvar = f_cvar[k]
            weights[k] = self._priors[k] * multivariate_normal.pdf(x, c_mean, c_cvar)
        
        # Normalized weights
        return weights / (np.sum(weights, axis=0) + sys.float_info.epsilon)

    def predict(self, x: np.array, d_given):
        """Infers the remaining state at the partially defined point x.

        Args:
            x (np.array): Partial point to infer remaining state at.
            d_given (iterable): Dimensions of x in order.

        Raises:
            Exception: Raised when dimensionality of x and the type hint do not match.

        Returns:
            np.array: Inferred remaining state at x.
        """
        x = x if x.ndim == 2 else x.reshape((1, x.size))

        if len(d_given) != x.shape[1]:
            raise Exception(f'Data with {x.shape[1]} dimensions was given, '
                            f'but {len(d_given)} dimensions were specified.')
        
        d_predicted = [i for i in range(self._means.shape[1]) if i not in d_given]

        # (K, num x)
        weights  = self.get_weights(x, d_given)
        p_mean   = np.zeros((x.shape[0], len(d_predicted)))
        # (given, given, K)
        f_g_cvar  = self._cvar.take(d_given, axis=1).take(d_given, axis=2)
        # (predicted, given, K)
        f_pg_cvar = self._cvar.take(d_predicted, axis=1).take(d_given, axis=2)
        for k in range(self._priors.size):
            # (given, 1)
            c_g_mean  = self._means[k].take(d_given)
            # (predicted, 1)
            c_p_mean  = self._means[k].take(d_predicted)
            # (given, given)
            c_g_cvar  = f_g_cvar[k]
            # (predicted, given)
            c_pg_cvar = f_pg_cvar[k]
            #     (pred, K)  (pred, given)  (given, given)               (num_x, given)
            aux_1 = (x - c_g_mean).T
            aux_2 = c_pg_cvar.dot(np.linalg.pinv(c_g_cvar)).dot(aux_1)
            aux   = c_p_mean + aux_2.T
            p_mean += weights[k] * aux
        return p_mean

    @property
    @lru_cache(1)
    def _cvar_tril_idx(self):
        temp = np.vstack(np.tril_indices(self._cvar.shape[1]))
        return tuple(np.hstack([np.vstack(([i] * temp.shape[1], temp)) for i in range(self.n_priors)]))


    def update_gaussian(self, priors=None, mu=None, sigma=None):
        """Returns a new GMM updated with the given deltas

        Args:
            priors (iterable, optional): Delta for priors.
            mu (np.array, optional): Delta for means.
            sigma (np.array, optional): Delta for (co-)variances.

        Returns:
            GMM: An updated GMM
        """        
        if priors is not None:
            new_priors = np.maximum(self._priors + priors, 0)
            prior_norm = new_priors.sum()
            if prior_norm <= 1e-4:
                raise ValueError(f'Prior update decreased all priors to less than {1e-4}. That is numerically unstable.')

            new_priors /= prior_norm
        else:
            new_priors = self._priors
        
        if mu is not None:
            new_mu = self._means + mu.reshape(self._means.shape)
        else:
            new_mu = self._means

        if sigma is not None:
            tril_idx = self._cvar_tril_idx

            sigma_mat = np.zeros(self._cvar.shape)
            sigma_mat[tril_idx]   = sigma
            np.transpose(sigma_mat, [0, 2, 1])[tril_idx] = sigma

            new_sigma = self._cvar + sigma_mat
            for x in range(new_sigma.shape[0]):
                if not isPD(new_sigma[x]):
                    new_sigma[x] = nearestPD(new_sigma[x])
        else:
            new_sigma = self._cvar
        
        return type(self)(new_priors, new_mu, new_sigma)

    @classmethod
    def load_model(cls, path):
        if Path(path).is_file():
            model = np.load(path, allow_pickle=True).item()

            priors = model["priors"].squeeze()
            mu     = model["mu"]
            sigma  = model["sigma"]
            
            # Compatibility with older models
            mu     = mu.T    if len(priors) == mu.shape[1]     else mu
            sigma  = sigma.T if len(priors) == sigma.shape[-1] else sigma
            return cls(priors, mu, sigma)
        raise Exception(f'Path "{path}" does not exist')

    def save_model(self, path):
        np.save(path, {'priors': self._priors,  # num_gaussians
                       'mu'    : self._means,  # observation_size * 2, num_gaussians
                       'sigma' : self._cvar})

    def expand_model(self, dims, sigma, cvar=0):
        new_mu        = np.hstack((self._means, np.zeros((self._means.shape[0], dims))))
        cv_var_mat    = np.ones(self._cvar.shape[:2] + (dims,)) * cvar
        new_dim_sigma = np.ones((self._cvar.shape[0], dims, dims)) * cvar
        diag_idx      = np.diag_indices(dims)
        for d in range(self._cvar.shape[0]):
            new_dim_sigma[d][diag_idx] = sigma

        new_sigma = np.concatenate((np.concatenate((self._cvar, cv_var_mat.transpose((0, 2, 1))), 1), 
                                    np.concatenate((cv_var_mat, new_dim_sigma), 1)), 2)

        return type(self)(self._priors, new_mu, new_sigma)

    def pdf(self, points : np.ndarray):
        """Calculate the PDF of an array of points in the form (n, dim)

        Args:
            points (np.ndarray): Stacked array of points
        """
        # Inverted covariances (priors, dims, dims)
        inv_cov = np.stack([np.linalg.inv(sigma) for sigma in self._cvar], 0)
        # Delta from the means as (priors, dims, points)
        delta   = np.expand_dims(self._means, -1) - np.expand_dims(points.T, 0)
        # Should be (priors, dims, points)
        temp    = np.stack([ic.dot(sd) for ic, sd in zip(inv_cov, delta)], 0)
        # Dot with the original delta (priors, points)
        dot     = (delta * temp).sum(axis=1)
        e       = np.exp(-0.5 * dot)
        factor  = 1 / ((2 * np.pi)**(np.pi * 0.5) * np.abs(self._cvar)**0.5)
        # Should be (priors, points)
        probabilites = factor * e * self._priors
        # Marginalize over overall probabilities
        probabilites /= probabilites.sum(axis=0)
        return probabilites

    def pi(self):
        return self._priors.copy()

    def sigma(self, dims_in=None, dims_out=None):
        """Returns the covariance matrix for the given dimensions.

        Args:
            dims_in (Iterable, optional): Indices of the input dimensions. Defaults to None.
            dims_out (Iterable, optional): Indices of the dimensions w.r.t. variance of the input dimensions is of interest. Defaults to None.

        Returns:
            np.ndarray: Filtered covariance matrix (n_priors, |dims_out|, |dims_in|)
        """
        if dims_in is None and dims_out is None:
            return self._cvar.copy()
        
        dims_in  = dims_in  if dims_in  is not None else np.arange(self.n_dims)
        dims_out = dims_out if dims_out is not None else np.arange(self.n_dims)

        return self._cvar.take(dims_out, axis=1).take(dims_in, axis=2)

    def mu(self, dims=None):
        """Return the means of the given dimensions.

        Args:
            dims (Iterable, optional): Indices of the dimensions of interest. Defaults to None.

        Returns:
            np.ndarray: Filtered means (n_priors, |dims|)
        """
        return self._means.copy() if dims is None else self._means.take(dims, axis=1)


if __name__=='__main__':
    class GMM3DPredictZ(GMM):
        def __init__(self, priors, means, cvar):
            super().__init__(priors=priors, means=means, cvar=cvar)

        def predict(self, xy):
            return super().predict(xy, [0, 1])

        def predictX(self, yz):
            return super().predict(yz, [1, 2])

    g1 = GMM(priors=np.array([1, 1, 1]), 
             means=np.array([[1, 2, 3]]).T, 
             cvar=np.array([[[0.1]]]*3))
    w1 = g1.get_weights(np.array([[1, 2, 3]]).T)
    print(f'Weights for 1, 2, 3:\n{w1}')

    g2 = GMM3DPredictZ(priors=np.array([1,  1,  1]),
                       means=np.array([[1,  1,  1],
                                       [4,  5, -2],
                                       [9, -2,  4]]),
                       cvar=np.stack([np.eye(3)]*3, axis=0))
    
    w2 = g2.get_weights(np.array([[1,  1,  1],
                                  [4,  5, -2],
                                  [9, -2,  4]]))
    print(f'Weights for 3d case:\n{w2}')

    w3 = g2.get_weights(np.array([[1,  1],
                                  [4,  5],
                                  [9, -2]]), dims=[0, 1])
    print(f'Weights for 3d case without Z:\n{w3}')

    w4 = g2.get_weights(np.array([[6.5, 1.5, 1]]))
    print(f'3d point perfectly between 2 and 3:\n{w4}')

    p1 = g2.predict(np.array([[4, 5]]))
    print(f'Prediction for Z at (4, 5): {p1}')
    print(f'Prediction for Z at (9, 2): {g2.predict(np.array([[9, -2]]))}')