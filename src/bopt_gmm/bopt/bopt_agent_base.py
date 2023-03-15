import numpy as np

from ConfigSpace import Configuration, \
                        ConfigurationSpace, \
                        Float

from dataclasses import dataclass, field
from datetime    import datetime
from functools   import lru_cache
from itertools   import product
from skopt       import Optimizer
from omegaconf   import ListConfig, \
                        DictConfig


from smac        import HyperparameterOptimizationFacade, \
                        BlackBoxFacade, \
                        HyperbandFacade, \
                        Scenario
from smac.runhistory.dataclasses import TrialValue, \
                                        TrialInfo

from typing      import Callable, Any, Iterable, Tuple

import bopt_gmm.gmm as lib_gmm
from bopt_gmm.logging import LoggerBase


def base_success(prior_obs, posterior_obs, action, reward, done):
    return done and reward > 0

def no_op(x):
    return x


BOPT_TIME_SCALE = 'bopt_step'

@dataclass
class GMMOptConfig:
    prior_range      : float = 0.15
    mean_range       : float = 0.05
    sigma_range      : float = 0.0
    opt_dims         : list  = None

@dataclass
class BOPTAgentConfig(GMMOptConfig):
    f_success        : Callable[[Any, Any, Any, float, bool], bool] = base_success
    early_tell       : int   = 5
    late_tell        : int   = 100000
    reward_processor : str   = 'mean'    # mean, raw
    base_estimator   : str   = 'GP'
    initial_p_gen    : str   = 'random'
    n_initial_points : int   = 10
    acq_func         : str   = 'gp_hedge'
    acq_optimizer    : str   = 'auto'
    gripper_command  : float = 0.5
    max_training_steps : int = 100


class GMMOptAgent(object):
    def __init__(self, gmm, config):
        self.base_model = gmm
        self.config     = config
        self.model      = gmm
        self._e_cvar_params = None
        self._complex_update_type = 'eigen'

    @property
    @lru_cache(1)
    def config_space(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        if self.config.prior_range != 0.0:
            cs.add_hyperparameters([Float(w, (-self.config.prior_range, self.config.prior_range), default=0) for w in self._weight_params])
        if self.config.mean_range != 0.0:
            cs.add_hyperparameters([Float(mp, (-self.config.mean_range, self.config.mean_range), default=0) for mp in self._mean_params.keys()])
        if self.config.sigma_range != 0.0:
            cvar_params = self._cvar_params
            if self._e_cvar_params is not None:
                if self._complex_update_type == 'eigen':
                    cs.add_hyperparameters([Float(sp, (1 - self.config.sigma_range, 1 + self.config.sigma_range), default=1.0) for sp in sum(cvar_params.values(), [])])
                else:
                    cs.add_hyperparameters([Float(sp, (-self.config.sigma_range, self.config.sigma_range), default=0.0) if '|' in sp else 
                                            Float(sp, (1 - self.config.sigma_range, 1 + self.config.sigma_range), default=1.0) for sp in sum(cvar_params.values(), [])])
            else:
                cs.add_hyperparameters([Float(sp, (-self.config.sigma_range, self.config.sigma_range), default=0) for sp in cvar_params.keys()])
        return cs

    @property
    @lru_cache(1)
    def _weight_params(self):
        return [f'weight_{x}' for x in range(self.base_model.n_priors)]

    @property
    @lru_cache(1)
    def _mean_params(self):
        opt_dims = [d for d in self.config.opt_dims if '|' not in d] if self.config.opt_dims is not None else []
        config_dims = self.config.opt_dims
        if type(config_dims) in {DictConfig, dict}:
            config_dims = config_dims['means']

        if type(config_dims) in {list, tuple, ListConfig} and len(opt_dims) > 0:
            return dict(sum([sum([[(f'mean_{d}_{y}_{x}', (y, x)) for x in self.base_model.semantic_dims()[d]]
                                                                 for d in config_dims], []) 
                                                                 for y in range(self.base_model.n_priors)], []))
        else:
            pass
        return dict(sum([[(f'mean_{y}_{x}', (y, x)) for x in range(self.base_model.n_dims)] for y in range(self.base_model.n_priors)], []))

    @property
    @lru_cache(1)
    def _cvar_params(self):
        config_dims = self.config.opt_dims
        if type(config_dims) in {DictConfig, dict}:
            config_dims = config_dims['cvars']

        if type(config_dims) in {list, tuple, ListConfig} and len(self.config.opt_dims) > 0:
            out = {}
            
            vars  = [d for d in self.config.opt_dims if '|' not in d]
            cvars = [d.split('|') for d in self.config.opt_dims if '|' in d]

            for v in vars:
                # Generate lower triangle coords
                coords = sum([[(y, x) for x in self.base_model.semantic_dims()[v][:i+1]] for i, y in enumerate(self.base_model.semantic_dims()[v])], [])
                
                for d in range(self.base_model.n_priors):
                    out.update({f'var_{v}_{d}_{y}_{x}': (d, y, x) for y, x in coords})
            
            for (state, inf) in cvars:
                # Full NxM coords
                coords = sum([[(y, x) for x in self.base_model.semantic_dims()[state]] for y in self.base_model.semantic_dims()[inf]], [])

                for d in range(self.base_model.n_priors):
                    out.update({f'cvar_{state}|{inf}_{d}_{y}_{x}': (d, y, x) for y, x in coords})
            return out
        elif type(config_dims) in {dict, DictConfig}:
            self._complex_update_type = config_dims['type'] if 'type' in config_dims else 'eigen'

            e_cvar_params = {k: [f'eu_{i}_{k}' for i in range(self.base_model.n_priors)] for k in config_dims['unary']} if config_dims['unary'] is not None else {}

            if config_dims['nary'] is not None:
                for k in config_dims['nary']:
                    dims = max([len(self.base_model.semantic_dims()[d]) for d in k.split('|')]) if '|' in k else len(self.base_model.semantic_dims()[k])
                    e_cvar_params[k] = sum([[f'e_{i}_{k}_{x}' for x in range(dims)] for i in range(self.base_model.n_priors)], [])
            
            if len(e_cvar_params) > 0:
                self._e_cvar_params = e_cvar_params
                return self._e_cvar_params
        return dict(sum([(f'cvar_{d}_{y}_{x}', (d, y, x)) for y, x in zip(*np.tril_indices(self.base_model.n_dims))], []))

    def update_model(self, parameter_update):
        if type(parameter_update) == TrialInfo:
            parameter_update = parameter_update.config

        start_idx = 0
        if self.config.prior_range != 0.0:
            if type(parameter_update) not in {dict, Configuration}:
                u_priors   = parameter_update[start_idx:self.base_model.n_priors]
                start_idx += self.base_model.n_priors
            else:
                u_priors   = np.asarray([parameter_update[p] for p in self._weight_params])
        else:
            u_priors = None

        if self.config.mean_range != 0.0:
            mu         = self.base_model.mu()
            mu_space   = mu.max(axis=0) - mu.min(axis=0)
            if type(parameter_update) not in {dict, Configuration}:
                u_mean     = np.asarray(parameter_update[start_idx:start_idx + mu.size]).reshape(mu.shape) * mu_space
                start_idx += mu.size
            else:
                u_mean = np.zeros(mu.shape)
                for k, coords in self._mean_params.items():
                    u_mean[coords] = parameter_update[k]
                # u_mean *= mu_space
        else:
            u_mean = None

        u_sigma_e = None
        u_sigma   = None
        if self.config.sigma_range != 0.0:
            if self._e_cvar_params is None:
                sigma       = self.base_model.sigma()
                sigma_space = sigma.max(axis=0) - sigma.min(axis=0)
                unit_update = np.zeros_like(sigma)
                if type(parameter_update) not in {dict, Configuration}:
                    unit_update[self._cvar_indices[0], 
                                self._cvar_indices[1], 
                                self._cvar_indices[2]] = parameter_update[start_idx:start_idx + len(self._cvar_indices) * len(self._cvar_indices[0])]
                    unit_update = np.transpose(unit_update, [0, 2, 1])
                else:
                    for n, c in self._cvar_params.items():
                        unit_update[c] = parameter_update[n]
                
                unit_update *= sigma_space

                # Transpose to accomodate lower triangle indices
                u_sigma = unit_update[self.base_model._cvar_tril_idx]
            else:
                u_sigma_e = {k: np.array([parameter_update[v] for v in p]).reshape((self.base_model.n_priors, 
                                                                                    len(p) // self.base_model.n_priors))
                                                              for k, p in self._e_cvar_params.items()}
        if self._complex_update_type == 'eigen':
            self.model = self.base_model.update_gaussian(u_priors, u_mean, u_sigma, sigma_eigen_update=u_sigma_e)
        else:
            self.model = self.base_model.update_gaussian(u_priors, u_mean, u_sigma, sigma_rotation=u_sigma_e)

    def reset(self):
        self.model = self.base_model

    def predict(self, observation):
        return self.model.predict(observation).flatten()


class BOPTGMMAgentBase(GMMOptAgent):
    @dataclass
    class BOPTState:
        step    : int   = 0
        updates : int   = 0
        reward  : float = 0.0
        reward_samples : int = 0
    
    @dataclass
    class State:
        n_step               : int  = 0
        trajectories         : list = field(default_factory=lambda: [[]]) 
        success_trajectories : list = field(default_factory=list) 
        gp_optimizer         : Any  = None
        current_update       : Any  = None
        bopt_state           : Any  = None
        obs_transform        : Any  = no_op


    def __init__(self, base_gmm, config: BOPTAgentConfig, obs_transform = no_op, logger : LoggerBase=None) -> None:
        super().__init__(base_gmm, config)
        self.logger     = logger

        self.state = BOPTGMMAgentBase.State(obs_transform=obs_transform)

        if self.logger is not None:
            self.logger.define_metric('bopt_x', BOPT_TIME_SCALE)
            self.logger.define_metric('bopt_y', BOPT_TIME_SCALE)

    def predict(self, observation):
        observation = self.state.obs_transform(observation)
        return {'motion': super().predict(observation), 'gripper': self.config.gripper_command}

    def step(self, prior_obs, posterior_obs, action, reward, done):
        transition = (prior_obs, posterior_obs, action, reward, done)
        self.state.n_step += 1
        self.state.trajectories[-1].append(transition)
        if self.config.f_success(*transition):
            self.state.success_trajectories.append(self.state.trajectories[-1])
            # print(f'Collected a successful trajectory. Now got {len(self.state.success_trajectories)}')
        if done:
            self.state.trajectories.append([])
            
            # If GP-Optimization is in progress, step it!
            if self.state.current_update is not None:
                self.step_optimizer(reward)

    def init_optimizer(self, base_accuracy=None):
        self.state.bopt_state    = BOPTGMMAgentBase.BOPTState()
        self.state.base_accuracy = base_accuracy if base_accuracy is not None else len(self.state.success_trajectories) / len(self.state.trajectories)

        # optim_params = []
        # if self.config.prior_range != 0.0:
        #     optim_params += [(-self.config.prior_range, self.config.prior_range)] * self.base_model.n_priors
        # if self.config.mean_range != 0.0:
        #     optim_params += [(-self.config.mean_range, self.config.mean_range)] * self.base_model.n_priors * self.base_model.n_dims
        # if self.config.sigma_range != 0.0:
        #     optim_params += [(-self.config.sigma_range, self.config.sigma_range)] * (len(self.base_model.state_dim) * len(self.base_model.prediction_dim) * self.base_model.n_priors)
        #     self._cvar_indices = list(zip(*product(range(self.base_model.n_priors), self.base_model.state_dim, self.base_model.prediction_dim)))

        # SKOPT OPTIMIZER
        # self.state.gp_optimizer  = Optimizer(optim_params,
        #                                      base_estimator=self.config.base_estimator,
        #                                      n_initial_points=self.config.n_initial_points,
        #                                      initial_point_generator=self.config.initial_p_gen,
        #                                      acq_func=self.config.acq_func,
        #                                      acq_optimizer=self.config.acq_optimizer)
        
        # SMAC
        self._scenario = Scenario(self.config_space, n_trials=self.config.max_training_steps)


        facade = {'hpo': HyperparameterOptimizationFacade,
                  'hb' : HyperbandFacade,
                  'bb' : BlackBoxFacade}[self.config.acq_optimizer]
        
        # intensifier = facade.get_intensifier(
        #     self._scenario,
        #     max_config_calls=1,  # We basically use one seed per config only
        # )

        self.state.gp_optimizer = facade(self._scenario,
                                         lambda config, seed, budget : 0,
                                         overwrite=True)
        # print(f'Base Model:\nPriors: {self.base_model.pi()}\nMu: {self.base_model.mu()}')

        self.update_model()

    def step_optimizer(self, reward):
        self.state.bopt_state.step   += 1
        self.state.bopt_state.reward += reward
        self.state.bopt_state.reward_samples += 1

        # if True:
        if self.state.bopt_state.step % min(max(int(1.0 / self.state.base_accuracy), self.config.early_tell), self.config.late_tell) == 0:
            print(f"Finished gp run {self.state.bopt_state.updates} ({self.state.bopt_state.step}). Return: {self.state.bopt_state.reward}. Let's go again!")
            self._tell(self.state.current_update, self.state.bopt_state.reward)    
            self.update_model()

    def update_model(self):
        for x in range(100):
            try:
                self.state.current_update = self.state.gp_optimizer.ask()

                super().update_model(self.state.current_update)
                u_sigma = self.model.sigma() - self.base_model.sigma()
                # print(f'Updated Model:\nPriors: {self.model.pi()}\nMu: {self.model.mu()}\nSigma-Delta: {u_sigma}')
                break
            except ValueError as e:
                pass
                # self.state.gp_optimizer.tell(self.state.current_update, 0)
                # self.state.bopt_state.updates += 1
        else:
            raise Exception(f'Repeated Bayesian Updates have failed to produce a valid update')

    def _tell(self, state, reward):
        reward = reward if self.config.reward_processor == 'raw' else reward / self.state.bopt_state.reward_samples
        # SKOPT
        # self.state.gp_optimizer.tell(state, -reward)

        reward = 100 - reward

        # SMAC
        self.state.gp_optimizer.tell(state, TrialValue(reward))

        self.state.bopt_state.updates += 1
        self.state.bopt_state.reward   = 0.0
        self.state.bopt_state.reward_samples = 0
        if self.logger is not None:
            self.logger.log({
                BOPT_TIME_SCALE: self.state.bopt_state.updates,
                'bopt_y': reward,
        #         'bopt_x': self.state.gp_optimizer.Xi[-1]
            })

    def reset(self):
        self.state = type(self.state)()
        self.model = self.base_model

    def has_gp_stage(self):
        return True

    def is_in_gp_stage(self):
        return self.state.gp_optimizer is not None

    def get_bopt_step(self):
        return self.state.bopt_state.updates if self.state.bopt_state is not None else 0
