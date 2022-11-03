from .bopt_agent_base import BOPTGMMAgentBase, \
                             BOPTAgentConfig

@dataclass
class BOPTAgentGenGMMConfig(BOPTAgentConfig):
    n_successes     : int = 10
    f_gen_gmm       : Callable[[Iterable[Tuple[Any, Any, Any, float, bool]], float], GMM] = base_gen_gmm
    delta_t         : float = 0.05
    debug_data_path : str = None
    debug_gmm_path  : str = None
    normalize_force : bool = False


class BOPTGMMCollectAndOptAgent(BOPTGMMAgentBase):
    
    def step(self, prior_obs, posterior_obs, action, reward, done):
        super().step(prior_obs, posterior_obs, action, reward, done)

        # Start optimization process
        if len(self.state.success_trajectories) == self.config.n_successes and not self.is_in_gp_stage():
            print('Starting Bayesian optimization')
        
            stamp = datetime.now()

            if self.config.debug_data_path is not None:
                np.savez(f'{self.config.debug_data_path}_{stamp}.npz', self.state.success_trajectories)
                print(f'Saved success trajectories to "{self.config.debug_data_path}"')

            if self.config.normalize_force:
                f_norm = BOPTGMMCollectAndOptAgent.calculate_force_normalization(self.state.success_trajectories)

                def normalize_obs(obs):
                    out = obs.copy()
                    out['force'] = obs['force'] * f_norm
                    return out

                self.state.obs_transform = normalize_obs

                print(f'Calculated {f_norm} as normalization factor for forces')
                self.state.success_trajectories = BOPTGMMCollectAndOptAgent.normalize_force_trajectories(f_norm, self.state.success_trajectories)

            self.base_model = self.config.f_gen_gmm(self.state.success_trajectories, self.config.delta_t)
            
            if self.config.debug_gmm_path is not None:
                self.base_model.save_model(f'{self.config.debug_gmm_path}_{stamp}.npy')

            self.init_optimizer()
            

    @staticmethod
    def calculate_force_normalization(trajectories):
        pos_force = np.abs(np.vstack([np.vstack([np.hstack((p['position'], p['force'])) for p, _, _, _, _ in t]) for t in trajectories])).max(axis=0)
        
        return (pos_force[:3] / pos_force[3:]).max()
    
    @staticmethod
    def normalize_force_trajectories(factor, trajectories):
        out = []
        for t in trajectories:
            nt = []
            for prior, posterior, action, reward, done in t:
                prior = {'position': prior['position'],
                            'force': prior['force'] * factor}
                posterior = {'position': posterior['position'],
                                'force': posterior['force'] * factor}
                nt.append((prior, posterior, action, reward, done))
            out.append(nt)
        return out

# ------ SINGELTON SEDS GENERATOR -------
SEDS = None

def seds_gmm_generator(seds_path, gmm_type, n_priors, objective='likelihood', tol_cutting=0.1, max_iter=600):
    # Ugly, I know
    global SEDS
    
    if SEDS is None:
        SEDS = SEDS_MATLAB(seds_path)

    def generator(transitions, delta_t):
        x0s, xTs, data, oIdx = gen_seds_data_from_transitions(transitions, delta_t, True)
        x0, xT, data, oIdx   = seds_data_prep_last_step(x0s, xTs, data, oIdx)
        gmm = SEDS.fit_model(x0, xT, data, oIdx, n_priors, 
                             objective=objective, dt=delta_t, 
                             tol_cutting=tol_cutting, max_iter=max_iter, gmm_type=gmm_type)
        return gmm
    
    return generator