import sys
import numpy as np

from argparse    import ArgumentError
from functools   import lru_cache
from itertools   import cycle, \
                        product
from pathlib     import Path
from random      import random
from scipy.stats import multivariate_normal
from typing      import Union

from prime_bullet import Quaternion

from .utils import isPD, nearestPD


GMM_MODEL_REGISTRY = {}

def add_gmm_model(cls, name=None):
    name = str(cls) if name is None else name
    GMM_MODEL_REGISTRY[name] = cls


class GMM(object):
    def __init__(self, priors: Union[int, np.array]=None, means: Union[int, np.array]=None, cvar: np.array=None, general_scale=1.0):
        # 1D - Simply the priors
        n_components = priors if type(priors) == int else len(priors)
        if type(priors) == int:
            self._priors = np.ones(n_components) / n_components
        elif priors.shape != (n_components, ):
            raise Exception(f'Expected priors to be of shape ({n_components},) '
                            f'but they are of shape {priors.shape}')
        elif np.sum(priors) <= 5 * sys.float_info.epsilon:
            raise ValueError(f'Value of priors is too low to normalize: {priors}')
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
        
        self._general_scale = general_scale
        self._GMM_TYPE = str(type(self))

    @property
    def n_priors(self):
        return len(self._priors)

    @property
    def n_dims(self):
        return self._means.shape[1]

    def sub_pdf(self, X, component, dims=None):
        _, n_features = X.shape
        mean    = self._mu(dims)[component]
        inv_cov = self._inv_sigma(dims)[component]
        det_cov = self._sigma_det(dims, dims)[component]

        if det_cov < 0:
            return 0

        delta = (X - mean)

        return np.exp(-0.5 * (delta * inv_cov.dot(delta.T).T).sum(axis=1)) / np.sqrt(((2 * np.pi) ** n_features) * det_cov)

    def get_weights(self, X: np.array, dims=None):
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
            if X.shape[1] < self._means.shape[1]: # Todo: Check fails for 1d inputs
                raise Exception(f'Dimensions for weighting need to '
                                f'be specified explicitly when given'
                                f'data is of lower dimensionality.')
                
            dims = range(self._means.shape[1])
        
        X = X if X.ndim == 2 else X.reshape((1, X.size))

        weights = np.zeros((self._priors.size, X.shape[0]))
        for k in range(self.n_priors):
            weights[k] = self._priors[k] * self.sub_pdf(X, k, dims)

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

        x = x * self._general_scale

        if len(d_given) != x.shape[1]:
            raise Exception(f'Data with {x.shape[1]} dimensions was given, '
                            f'but {len(d_given)} dimensions were specified.')
        
        d_predicted = tuple(i for i in range(self._means.shape[1]) if i not in d_given)

        # (K, num x)
        weights  = self.get_weights(x, d_given)
        p_mean   = np.zeros((x.shape[0], len(d_predicted)))
        # (given, given, K)
        f_g_cvar  = self._sigma(d_given, d_given)
        # (predicted, given, K)
        f_pg_cvar = self._sigma(d_given, d_predicted)
        for k, (c_g_mean, c_p_mean, c_g_cvar, c_pg_cvar) in enumerate(zip(self._mu(d_given), 
                                                                          self._mu(d_predicted), f_g_cvar, f_pg_cvar)):
            #     (pred, K)  (pred, given)  (given, given)               (num_x, given)
            divergence = (x - c_g_mean).T
            aux_2 = c_pg_cvar.dot(np.linalg.pinv(c_g_cvar)).dot(divergence)
            aux   = c_p_mean + aux_2.T
            p_mean += (aux.T * weights[k]).T

        return p_mean / self._general_scale

    def conditional_pdf(self, X : np.array, d_given, *Ys : np.array):
        X = X if X.ndim == 2 else x.reshape((1, X.size))

        X = X * self._general_scale

        if len(d_given) != X.shape[1]:
            raise Exception(f'Data with {X.shape[1]} dimensions was given, '
                            f'but {len(d_given)} dimensions were specified.')

        if len(Ys) == 0:
            raise Exception('Need Y points to assess conditional probability at')

        d_predicted = tuple(i for i in range(self._means.shape[1]) if i not in d_given)
        Y = np.stack(Ys, 1)
        if max([Y.shape[1] != len(d_predicted) for Y in Ys]):
            raise Exception(f'Inference points do not match desidered dimensionality {len(d_predicted)} {[Y.shape[1] for Y in Ys]}')
        
        if max([Y.shape[0] != X.shape[0] for Y in Ys]):
            raise Exception(f'Inference points must match number of sample points {X.shape[0]} {[Y.shape[0] for Y in Ys]}')

        weights     = self.get_weights(X, d_given).T
        g_mean      = self._mu(d_given)
        p_mean      = self._mu(d_predicted)
        # (given, given, K)
        f_g_cvar    = self._sigma(d_given, d_given)
        # (predicted, given, K)
        f_pg_cvar   = self._sigma(d_given, d_predicted)
        # (predicted, predicted, K)
        f_pp_cvar   = self._sigma(d_predicted, d_predicted)
        # (given, predicted, K)

        pdfs = np.zeros((Y.shape[0], Y.shape[1]))

        f_g_inv_cvar = self._inv_sigma(d_given)
        f_gp_cvar    = self._sigma(d_given, d_predicted)
        f_pg_cvar    = self._sigma(d_predicted, d_given)
        f_p_cvar     = self._sigma(d_predicted, d_predicted)

        cd_cvar_temp = np.stack([gp_cvar.dot(inv_cvar) for gp_cvar, inv_cvar in zip(f_gp_cvar, f_g_inv_cvar)], 0)

        cd_cvar = np.stack([p_cvar - t_cvar.dot(pg_cvar) for p_cvar, t_cvar, pg_cvar in zip(f_p_cvar, cd_cvar_temp, f_pg_cvar)], 0)

        for k, (x, w, y) in enumerate(zip(X, weights, Y)):
            g_delta = (x - g_mean)
            means   = p_mean + np.vstack([t_cvar.dot(d) for t_cvar, d in zip(cd_cvar_temp, g_delta)])

            sub_gmm = GMM(w, means, cd_cvar)
            pdfs[k] = sub_gmm.pdf(y)

        return pdfs.T

    @property
    @lru_cache(1)
    def _cvar_tril_idx(self):
        temp = np.vstack(np.tril_indices(self._cvar.shape[1]))
        return tuple(np.hstack([np.vstack(([i] * temp.shape[1], temp)) for i in range(self.n_priors)]))

    def update_gaussian(self, priors=None, mu=None, sigma=None, sigma_scale=None, sigma_eigen_update=None, sigma_rotation=None):
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
            if sigma.shape == self._cvar.shape:
                sigma_mat = sigma
            else:  # 1-D lower triangle update            
                tril_idx = self._cvar_tril_idx

                sigma_mat = np.zeros(self._cvar.shape)
                sigma_mat[tril_idx]   = sigma
                np.transpose(sigma_mat, [0, 2, 1])[tril_idx] = sigma

            new_sigma = self._cvar + sigma_mat
            for x in range(new_sigma.shape[0]):
                if not isPD(new_sigma[x]):
                    new_sigma[x] = nearestPD(new_sigma[x])
        else:
            new_sigma = self._cvar.copy()
        
        if sigma_scale is not None:
            new_sigma = new_sigma * sigma_scale

        if sigma_eigen_update is not None:
            for k, update in sigma_eigen_update.items():
                if '|' not in k or k == 'STATE':
                    k_dim = self.semantic_dims()[k] if k != 'STATE' else self.state_dim
                    if update.shape != (self.n_priors, len(k_dim)) and update.shape != (self.n_priors, 1):
                        raise Exception(f'Expected sigma eigenval update for {k} to have shape {(self.n_priors, len(k_dim))}, or {(self.n_priors, 1)}, but got {update.shape}')

                    coords = tuple(zip(*product(k_dim, k_dim)))

                    for k, (u, sigma_k) in enumerate(zip(update, self.sigma(k_dim, k_dim))):
                        w, v = np.linalg.eig(sigma_k)
                        new_sigma_k = v.dot(np.diag(w * u)).dot(np.linalg.inv(v))

                        new_sigma[k, coords[0], coords[1]] = new_sigma_k.flatten()
                else:
                    dim_in, dim_out = k.split('|')
                    dim_in  = self.semantic_dims()[dim_in]  if dim_in != 'STATE'  else self.state_dim
                    dim_out = self.semantic_dims()[dim_out] if dim_in != 'ACTION' else self.prediction_dim

                    if update.shape != (self.n_priors, len(dim_in)) and update.shape != (self.n_priors, 1):
                        raise Exception(f'Expected sigma eigenval update for {k} to have shape {(self.n_priors, len(dim_in))}, or {(self.n_priors, 1)}, but got {update.shape}')

                    if len(dim_in) != len(dim_out):
                        raise NotImplementedError(f'Currently eigenvalue updates are only possible for square associations. Association {k} is non-square.')

                    coords = tuple(zip(*product(dim_out, dim_in)))

                    for k, (u, sigma_k) in enumerate(zip(update, self.sigma(dim_in, dim_out))):
                        w, v = np.linalg.eig(sigma_k)
                        new_sigma_k = v.dot(np.diag(w * u)).dot(np.linalg.inv(v))

                        new_sigma[k, coords[0], coords[1]] = new_sigma_k.flatten()
                        # Set the upper triangle part
                        new_sigma[k, coords[1], coords[0]] = new_sigma_k.T.flatten()

        if sigma_rotation is not None:
            for k, update in sigma_rotation.items():
                if '|' not in k:
                    k_dim = self.semantic_dims()[k]
                    if update.shape != (self.n_priors, len(k_dim)) and update.shape != (self.n_priors, 1):
                        raise Exception(f'Expected sigma eigenval update for {k} to have shape {(self.n_priors, len(k_dim))}, or {(self.n_priors, 1)}, but got {update.shape}')

                    coords = tuple(zip(*product(k_dim, k_dim)))

                    for k, (u, sigma_k) in enumerate(zip(update, self.sigma(k_dim, k_dim))):
                        w, v = np.linalg.eig(sigma_k)
                        new_sigma_k = v.dot(np.diag(w * u)).dot(np.linalg.inv(v))

                        new_sigma[k, coords[0], coords[1]] = new_sigma_k.flatten()
                else:
                    dim_in, dim_out = k.split('|')
                    dim_in  = self.semantic_dims()[dim_in]
                    dim_out = self.semantic_dims()[dim_out]

                    if update.shape != (self.n_priors, len(dim_in)) and update.shape != (self.n_priors, 1):
                        raise Exception(f'Expected sigma eigenval update for {k} to have shape {(self.n_priors, len(dim_in))}, or {(self.n_priors, 1)}, but got {update.shape}')

                    if len(dim_in) != len(dim_out):
                        raise NotImplementedError(f'Currently eigenvalue updates are only possible for square associations. Association {k} is non-square.')

                    coords = tuple(zip(*product(dim_out, dim_in)))

                    for k, (u, sigma_k) in enumerate(zip(update, self.sigma(dim_in, dim_out))):
                        rot = Quaternion.from_euler(*u).matrix()

                        new_sigma_k = rot.dot(sigma_k)
                        new_sigma[k, coords[0], coords[1]] = new_sigma_k.flatten()
                        # Set the upper triangle part
                        new_sigma[k, coords[1], coords[0]] = new_sigma_k.T.flatten()

        return type(self)(new_priors, new_mu, new_sigma)

    @lru_cache(1)
    def semantic_inference_weights(self, dims):
        """Returns the relative importance of the semantic dimensions 
           for calculating the weighting of components given a sample.

           Specifically:
            out = {dim_0: [c_0_0, ..., c_0_k],
                   ...
                   dim_m: [c_m_0, ..., c_m_k]}
            
            where sum(c_0_i, ..., c_m_i) == 1
        """
        # (d, k) matrix of determinants
        out = np.asarray([[np.linalg.det(i_sigma_d) for i_sigma_d in self._inv_sigma(self.semantic_dims()[d])] 
                                                    for d in dims])

        # Normalize
        out /= out.sum(axis=0)
        return out.T

    @lru_cache(10)
    def semantic_prediction_weights(self, p_dims, given_dims):
        """Returns the relative importance of the semantic dimensions 
           for calculating the action of components given a sample.

           Specifically:
            out = {dim_0: [c_0_0, ..., c_0_k],
                   ...
                   dim_m: [c_m_0, ..., c_m_k]}
            
            where sum(c_0_i, ..., c_m_i) == 1
        """
        out = {}
        for p in p_dims:
            try:
                out[p] = np.abs(np.asarray([[1 / np.abs(sigma_dp).sum()
                                             for sigma_dp in self._sigma(self.semantic_dims()[d], 
                                                                        self.semantic_dims()[p])] 
                                             for d in given_dims]))
            except KeyError as e:
                raise Exception(f'Unknown semantic dimension "{k}"')

            # Normalize
            out[p] /= out[p].sum(axis=0)
            out[p] = out[p].T
        return out

    def calculate_reweighting_inference_update(self, target, dims):
        """Given a desired target weight distribution, calculates 
           an update scaling matrix to achieve this distribution."""
        current_weights = self.semantic_inference_weights(dims)

        update_factors = {k: t / c for k, t, c in zip(dims, target.T, current_weights.T)}

        update_matrix  = np.ones_like(self._cvar)
        for k, fs in update_factors.items():
            k_dim  = self.semantic_dims()[k]
            coords = list(zip(*product(k_dim, k_dim)))

            for k, f in enumerate(fs):
                update_matrix[k, coords[0], coords[1]] = f**(1/len(k_dim))  # Actual factor is the nth root
        return update_matrix

    def calculate_reweighting_prediction_update(self, target, dims_in):
        """Given a desired target weight distribution, calculates 
           an update scaling matrix to achieve this distribution."""
        current_weights = self.semantic_prediction_weights(tuple(target.keys()), tuple(dims_in))

        update_factors = {k: t / current_weights[k] for k, t in target.items()}

        update_matrix  = np.ones_like(self._cvar)
        for k, fs in update_factors.items():
            k_dim = self.semantic_dims()[k]
            for d, f in zip(dims_in, fs):
                update_matrix[:, k_dim, self.semantic_dims()[d]] = f
        return update_matrix

    @classmethod
    def load_model(cls, path):
        if Path(path).is_file():
            model = np.load(path, allow_pickle=True).item()

            priors = model["priors"].squeeze()
            mu     = model["mu"]
            sigma  = model["sigma"]
            gen_scale = model["scale"] if "scale" in model else 1.0
            
            if 'type' in model:
                typ = GMM_MODEL_REGISTRY[model["type"]]
            else:  # Maintain compatibility to old models
                typ = cls
            custom_data = model["custom_data"]
            
            # Compatibility with older models
            mu     = mu.T    if len(priors) == mu.shape[1]     else mu
            sigma  = sigma.T if len(priors) == sigma.shape[-1] else sigma
            return typ(priors, mu, sigma, general_scale=gen_scale, **custom_data)
        raise Exception(f'Path "{path}" does not exist')

    def save_model(self, path):
        np.save(path, {'priors': self._priors,  # num_gaussians
                       'mu'    : self._means,  # observation_size * 2, num_gaussians
                       'sigma' : self._cvar,
                       'scale' : self._general_scale,
                       'type'  : self._GMM_TYPE,
                       'custom_data': self._custom_data()})

    def _custom_data(self):
        """Returns a dictionary of additional data needed to instantiate this type of GMM"""
        return {}

    def expand_model(self, dims, sigma, cvar=0):
        new_mu        = np.hstack((self._means, np.zeros((self._means.shape[0], dims))))
        cv_var_mat    = np.ones(self._cvar.shape[:2] + (dims,)) * cvar
        new_dim_sigma = np.ones((self._cvar.shape[0], dims, dims)) * cvar
        diag_idx      = np.diag_indices(dims)
        for d in range(self._cvar.shape[0]):
            new_dim_sigma[d][diag_idx] = sigma

        new_sigma = np.concatenate((np.concatenate((self._cvar, cv_var_mat.transpose((0, 2, 1))), 1), 
                                    np.concatenate((cv_var_mat, new_dim_sigma), 1)), 2)

        return type(self)(self._priors, new_mu, new_sigma, general_scale=self._general_scale)

    @lru_cache(10)
    def _mu(self, dims=None):
        if dims is not None:
            return self._means[:,dims]
        return self._means

    @lru_cache(10)
    def _sigma(self, dims_in=None, dims_out=None):
        if dims_in is not None:
            if dims_out is None:
                return self._cvar[:, :, dims_in]
            return self._cvar[:, dims_out][:,:,dims_in]
        if dims_out is not None:
            return self._cvar[:, dims_out, :]
        return self._cvar

    @lru_cache(10)
    def _sigma_det(self, dims_in=None, dims_out=None):
        sigma = self._sigma(dims_in, dims_out)
        return np.asarray([np.linalg.det(s) for s in sigma])

    @lru_cache(10)
    def _inv_sigma(self, dims=None):
        return np.stack([np.linalg.inv(sigma) for sigma in self._sigma(dims, dims)], 0)

    def pdf(self, points : np.ndarray, dims=None):
        """Calculate the PDF of an array of points in the form (n, dim)

        Args:
            points (np.ndarray): Stacked array of points
        """
        pdf = np.zeros(points.shape[0])
        for k, w in enumerate(self._priors):
            pdf += w * self.sub_pdf(points, k, dims)

        return pdf

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

    def semantic_dims(self):
        raise NotImplementedError

    def semantic_obs_dims(self):
        raise NotImplementedError

add_gmm_model(GMM)


def gen_prior_gmm(new_type, gmm, model_kwargs={}):
    prior_gmm = new_type(gmm.pi(), **model_kwargs)

    for d, i in gmm.semantic_dims().items():
        if d in prior_gmm.semantic_dims():
            # Assign means
            coords_base = list(zip(*product(range(gmm.n_priors), i)))
            coords_new  = list(zip(*product(range(gmm.n_priors), prior_gmm.semantic_dims()[d])))
            prior_gmm._means[coords_new[0], coords_new[1]] = gmm._means[coords_base[0], coords_base[1]]

            # Copy variances and covariances
            for d2, i2 in gmm.semantic_dims().items():
                if d2 in prior_gmm.semantic_dims():
                    coords_base = list(zip(*product(range(gmm.n_priors), i2, i)))
                    coords_new  = list(zip(*product(range(gmm.n_priors), 
                                                    prior_gmm.semantic_dims()[d2],
                                                    prior_gmm.semantic_dims()[d])))
                    prior_gmm._cvar[coords_new[0], 
                                    coords_new[1], 
                                    coords_new[2]] = gmm._cvar[coords_base[0],
                                                               coords_base[1], 
                                                               coords_base[2]]
            
    return prior_gmm
