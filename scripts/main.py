import hydra
import numpy as np

from argparse                   import ArgumentParser
from ConfigSpace                import ConfigurationSpace
from ConfigSpace                import UniformFloatHyperparameter
from dataclasses                import dataclass
from datetime                   import datetime
from omegaconf                  import OmegaConf
from pathlib                    import Path
from skopt                      import gp_minimize
# from smac.facade.smac_bb_facade import SMAC4BB
from smac.scenario.scenario     import Scenario
from tqdm                       import tqdm

from bopt_gmm.bopt import BOPTGMMCollectAndOptAgent, \
                          BOPTGMMAgent,              \
                          BOPTAgentGMMConfig,        \
                          BOPTAgentGenGMMConfig,     \
                          BOPT_TIME_SCALE
from bopt_gmm.gmm import GMMCart3D, \
                         GMMCart3DForce, \
                         GMMCart3DTorque

from bopt_gmm.gmm.generation import seds_gmm_generator, \
                                    em_gmm_generator

from bopt_gmm.utils   import conf_checksum
from bopt_gmm.logging import WBLogger, \
                             BlankLogger, \
                             LivePlot, \
                             dpg

from bopt_gmm.envs import PegEnv, \
                          DoorEnv


class AgentWrapper(object):
    def __init__(self, model, force_norm=1.0, gripper_command=0.0) -> None:
        self._model = model
        self.pseudo_bopt_step = 0
        self._force_norm = force_norm
        self._gripper_command = gripper_command

    def predict(self, obs):
        if 'force' in obs:
            obs['force'] = obs['force'] * self._force_norm
        return {'motion': self._model.predict(obs).flatten(), 'gripper': self._gripper_command}

    def step(self, *args):
        pass
    
    def has_gp_stage(self):
        return False

    def is_in_gp_stage(self):
        return False

    def get_bopt_step(self):
        self.pseudo_bopt_step += 1
        return self.pseudo_bopt_step

def evaluate(env, model,
             max_steps=2000,
             num_episodes=10,
             show_force=False,
             render=False,
             force_norm=1.0,
             logger=None,
             const_gripper_cmd=0.0):
    return evaluate_agent(env, AgentWrapper(model, force_norm, const_gripper_command), max_steps, 
                          num_episodes, show_force, render, logger=logger)


@dataclass
class RunAccumulator:
    _steps     : int   = 0
    _reward    : float = 0.0
    _successes : int   = 0
    _runs      : int   = 0

    def log_run(self, steps, reward, success):
        self._runs      += 1
        self._steps    += steps
        self._reward    += reward
        self._successes += int(success)

    def get_stats(self):
        if self._runs == 0:
            return 0.0, 0
        return self._steps / self._runs, self._reward, self._successes / self._runs


def evaluate_agent(env, agent, max_steps=2000, num_episodes=10, show_force=False, 
                   render=False, logger=None, opt_model_dir=None, checkpoint_freq=10):
    successful_episodes, episodes_returns, episodes_lengths = 0, [], []
    agent_in_gp = False
    last_model_save = 0

    if logger is not None:
        logger.define_metric('bopt accuracy', BOPT_TIME_SCALE)
        logger.define_metric('bopt reward', BOPT_TIME_SCALE)
        logger.define_metric('bopt mean steps', BOPT_TIME_SCALE)

    if show_force:
        dpg.create_context()
        dpg.create_viewport(title='Live Vis', width=900, height=450)
        dpg.setup_dearpygui()
        live_plot = LivePlot('Forces', {'force_x': 'Force X', 
                                        'force_y': 'Force Y',
                                        'force_z': 'Force Z'})
        dpg.show_viewport()

    bopt_step = 0
    prev_bopt_step = 0
    ep_acc = RunAccumulator()

    with tqdm(total=num_episodes, desc="Evaluating model") as pbar:
        while bopt_step < num_episodes:
            bopt_step = agent.get_bopt_step() if agent.has_gp_stage() else prev_bopt_step + 1
            pbar.update(bopt_step - prev_bopt_step)
            if bopt_step != prev_bopt_step and agent.has_gp_stage():
                if logger is not None:
                    bopt_mean_steps, bopt_reward, bopt_accuracy = ep_acc.get_stats()
                    logger.log({'bopt mean steps': bopt_mean_steps,
                                'bopt reward'    : bopt_reward, 
                                'bopt accuracy'  : bopt_accuracy})

                ep_acc = RunAccumulator()

            prev_bopt_step = bopt_step

            observation = env.reset()
            episode_return = 0
            
            for step in range(max_steps):
                action = agent.predict(observation)
                # print(observation)
                post_observation, reward, done, info = env.step(action)
                episode_return += reward
                done = done or (step == max_steps - 1)

                if show_force and dpg.is_dearpygui_running():
                    live_plot.add_value('force_x', observation['force'][0])
                    live_plot.add_value('force_y', observation['force'][1])
                    live_plot.add_value('force_z', observation['force'][2])
                    dpg.render_dearpygui_frame()

                    # print(observation['force'])

                agent.step(observation, post_observation, action, episode_return, done)
                if agent.is_in_gp_stage() and not agent_in_gp:
                    if opt_model_dir is not None:
                        agent.base_model.save_model(f'{opt_model_dir}/gmm_base.npy')
                    agent_in_gp = True
                observation = post_observation
                if opt_model_dir is not None and bopt_step - last_model_save >= checkpoint_freq:
                    agent.model.save_model(f'{opt_model_dir}/gmm_{bopt_step}.npy')
                    last_model_save = bopt_step

                if render:
                    env.render()
                
                if done:
                    break

            if agent.is_in_gp_stage():
                ep_acc.log_run(step + 1, episode_return, info['success'])

            episodes_returns.append(episode_return)
            episodes_lengths.append(step)

            if info["success"]:
                successful_episodes += 1
                print(f'Number of successes: {successful_episodes}\nCurrent Accuracy: {successful_episodes / len(episodes_returns)}')

            accuracy = successful_episodes / len(episodes_returns)

            if logger is not None:
                logger.log({'accuracy': accuracy, 
                            'success': int(info['success']),
                            'reward': episode_return, 'steps': step + 1})

    if opt_model_dir is not None:
        agent.model.save_model(f'{opt_model_dir}/gmm_final.npy')

    if show_force:
        dpg.destroy_context()

    return accuracy, np.mean(episodes_returns), np.mean(episodes_lengths)


def opt_prior_and_means(base_gmm, env, prior_bounds=(-0.15, 0.15), mean_bounds=(-0.005, 0.005)):

    def gp_eval_func(gmm_update):
        gmm = base_gmm.update_gaussian(priors=np.asarray(gmm_update[:base_gmm.n_priors]), 
                                       mu=np.asarray(gmm_update[base_gmm.n_priors:]))
        
        accuracy, mean_return, mean_length = evaluate(env, gmm, max_steps=600, num_episodes=1)
        print("Accuracy:", accuracy, "mean_return:", mean_return)
        return -mean_return

    res = gp_minimize(gp_eval_func,
                      [prior_bounds]*base_gmm.n_priors + [mean_bounds] * base_gmm.n_priors * base_gmm.n_dims,
                      acq_func="EI",  # the acquisition function
                      n_calls=200,  # the number of evaluations of f
                      n_random_starts=20,  # the number of random initialization points
                      noise="gaussian",  # the objective returns noisy gaussian observations
                      random_state=1234)
    return res


def opt_pos_f_means(base_gmm, env, bounds=(-5, 5)):

    def gp_eval_func(gmm_update):
        expanded_priors = np.zeros(base_gmm.n_priors)
        # expanded_priors[-2:] = gmm_update[:2]
        expanded_means  = np.zeros((base_gmm.n_priors, base_gmm.n_dims))
        expanded_means[:,-2:] = np.asarray(gmm_update).reshape((base_gmm.n_priors, 2)) 
        gmm = base_gmm.update_gaussian(expanded_priors, 
                                       expanded_means.flatten())
        
        accuracy, mean_return, mean_length = evaluate(env, gmm, max_steps=600, num_episodes=5)
        print("Accuracy:", accuracy, "mean_return:", mean_return)
        return -mean_return

    res = gp_minimize(gp_eval_func,
                      [bounds]*base_gmm.n_priors*2,
                      acq_func="EI",  # the acquisition function
                      n_calls=200,  # the number of evaluations of f
                      n_random_starts=20,  # the number of random initialization points
                      noise="gaussian",  # the objective returns noisy gaussian observations
                      random_state=1234)
    return res


# def opt_smac_prior_and_means(base_gmm, env, prior_bounds=(-0.15, 0.15), mean_bounds=(-0.005, 0.005)):
#     cspace = ConfigurationSpace()

#     prior_names = []
#     means_names = []

#     for x in range(base_gmm.n_priors):
#         prior_name = f'prior_{x}'
#         prior_names.append(prior_name)
#         cspace.add_hyperparameter(UniformFloatHyperparameter(prior_name, *prior_bounds))
#         for val in 'x y z vx vy vz'.split(' '):
#             name = f'mean_{x}_{val}'
#             means_names.append(name)
#             cspace.add_hyperparameter(UniformFloatHyperparameter(name, *mean_bounds))

#     scenario = Scenario({
#         'run_obj': 'quality',
#         'runcount-limit': 50,
#         'cs': cspace
#     })

#     def eval(config):
#         gmm = base_gmm.update_gaussian([config[n] for n in prior_names],
#                                        np.array([config[n] for n in means_names]))
#         accuracy, mean_return, mean_length = evaluate(env, gmm, max_steps=600, num_episodes=5)
#         print("Accuracy:", accuracy, "mean_return:", mean_return)
#         return -mean_return

#     smac = SMAC4BB(scenario=scenario, tae_runner=eval)
#     best_config = smac.optimize()

#     print(f'Best update:\nPriors: {[best_config[n] for n in prior_names]}\nMeans: {np.array([best_config[n] for n in means_names])}')


GMM_TYPES = {'position': GMMCart3D,
                'force': GMMCart3DForce,
               'torque': GMMCart3DTorque}

def load_gmm(gmm_config):
    return GMM_TYPES[gmm_config.type].load_model(gmm_config.model)


def main_bopt_agent(env, bopt_agent_config, conf_hash, show_force=True, wandb=False, log_prefix=None, model_dir=None):

    if bopt_agent_config.agent not in {'bopt-gmm', 'dbopt'}:
        raise Exception(f'Unkown agent type "{bopt_agent_config.agent}"')

    if model_dir is not None:
        model_dir = f'{model_dir}_{conf_hash}'
        p = Path(model_dir)
        if not p.exists():
            p.mkdir()

        with open(f'{model_dir}/config.yaml', 'w') as cfile:
            cfile.write(OmegaConf.to_yaml(bopt_agent_config))

    run_id = f'{bopt_agent_config.agent}_{conf_hash}'
    run_id = f'{log_prefix}_{run_id}' if log_prefix is not None else run_id

    logger = WBLogger('bopt-gmm', run_id, True) if wandb else BlankLogger()
    logger.log_config(bopt_agent_config)

    base_gmm = load_gmm(bopt_agent_config.gmm)

    if bopt_agent_config.agent == 'bopt-gmm':
        if bopt_agent_config.gmm_generator.type == 'seds':
            seds_config = bopt_agent_config.gmm_generator

            gmm_generator = seds_gmm_generator(seds_config.seds_path,
                                               GMMCart3DForce,
                                               seds_config.n_priors,
                                               seds_config.objective,
                                               seds_config.tol_cutting,
                                               seds_config.max_iter)
        else:
            raise Exception(f'Unknown GMM generator "{bopt_agent_config.gmm_generator}"')

        config = BOPTAgentGenGMMConfig(prior_range=bopt_agent_config.prior_range,
                                       mean_range=bopt_agent_config.mean_range,
                                       early_tell=bopt_agent_config.early_tell,
                                       late_tell=bopt_agent_config.late_tell,
                                       reward_processor=bopt_agent_config.reward_processor,
                                       n_successes=bopt_agent_config.n_successes,
                                       base_estimator=bopt_agent_config.base_estimator,
                                       initial_p_gen=bopt_agent_config.initial_p_gen,
                                       n_initial_points=bopt_agent_config.n_initial_points,
                                       acq_func=bopt_agent_config.acq_func,
                                       acq_optimizer=bopt_agent_config.acq_optimizer,
                                       gripper_command=bopt_agent_config.gripper_command,
                                       delta_t=env.dt,
                                       f_gen_gmm=gmm_generator,
                                       debug_data_path=f'{model_dir}/{bopt_agent_config.debug_data_path}',
                                       debug_gmm_path=f'{model_dir}/{bopt_agent_config.debug_gmm_path}')

        agent  = BOPTGMMCollectAndOptAgent(base_gmm, config, logger=logger)
    elif bopt_agent_config.agent == 'dbopt':
        config = BOPTAgentGMMConfig(prior_range=bopt_agent_config.prior_range,
                                    mean_range=bopt_agent_config.mean_range,
                                    early_tell=bopt_agent_config.early_tell,
                                    late_tell=bopt_agent_config.late_tell,
                                    reward_processor=bopt_agent_config.reward_processor,
                                    base_estimator=bopt_agent_config.base_estimator,
                                    initial_p_gen=bopt_agent_config.initial_p_gen,
                                    n_initial_points=bopt_agent_config.n_initial_points,
                                    acq_func=bopt_agent_config.acq_func,
                                    acq_optimizer=bopt_agent_config.acq_optimizer,
                                    gripper_command=bopt_agent_config.gripper_command,
                                    base_accuracy=bopt_agent_config.base_accuracy)

        agent  = BOPTGMMAgent(base_gmm, config, logger=logger)

    acc, return_mean, mean_ep_length = evaluate_agent(env, agent, max_steps=600, num_episodes=200, 
                                                      show_force=show_force, logger=logger, opt_model_dir=model_dir)
    print(f'Accuracy: {acc} Mean return: {return_mean} Mean ep length: {mean_ep_length}')
    bopt_res = agent.state.gp_optimizer.get_result()
    print(f'F means: {bopt_res.x}\nReward: {bopt_res.fun}')



if __name__ == '__main__':
    parser = ArgumentParser(description='Using hydra without losing control of your code.')
    parser.add_argument('hy', type=str, help='Hydra config to use. Relative to root of "config" dir')
    parser.add_argument('--overrides', default=[], type=str, nargs='*', help='Overrides for hydra config')
    parser.add_argument('--trajectories', default=None, help='Trajectories to fit a new GMM to')
    parser.add_argument('--mode', default='bopt-gmm', help='Modes to run the program in.', choices=['bopt-gmm', 'eval-gmm', 'vis'])
    parser.add_argument('--run-prefix', default=None, help='Prefix for the generated run-id for the logger')
    parser.add_argument('--wandb', action='store_true', help='Enable W&B logging.')
    parser.add_argument('--model-dir', default=None, help='Directory to save models and data to. Will be created if non-existent')
    args = parser.parse_args()

    # Point hydra to the root of your config dir. Here it's hard-coded, but you can also
    # use "MY_MODULE.__path__" to localize it relative to your python package
    hydra.initialize(config_path="../config")
    cfg = hydra.compose(args.hy, overrides=args.overrides)

    # from omegaconf import OmegaConf

    # print(OmegaConf.to_yaml(cfg))
    # exit()

    if args.trajectories is not None:
        trajs = next(iter(np.load(args.trajectories, allow_pickle=True).values()))

        f_norm = BOPTGMMCollectAndOptAgent.calculate_force_normalization(trajs)

        print(f'Trajectory normalization is {f_norm}')

        trajs = BOPTGMMCollectAndOptAgent.normalize_force_trajectories(f_norm, trajs)

        gmm_generator = seds_gmm_generator(cfg.seds_config.seds_path,
                                           GMMCart3DForce,
                                           cfg.seds_config.n_priors,
                                           cfg.seds_config.objective,
                                           cfg.seds_config.tol_cutting,
                                           cfg.seds_config.max_iter)
        gmm = gmm_generator(trajs, cfg.dt)
        gmm.save_model(f'last_model_{f_norm}')
        # report_gmm_seds_compliance(gmm, np.vstack([t[-1][0]['position'] for t in trajs]))

        env = PegEnv(cfg.env, cfg.show_gui)

        acc, returns, lengths = evaluate(env, gmm,
                                         max_steps=600,
                                         num_episodes=100,
                                         show_force=cfg.show_gui,
                                         force_norm=f_norm)
        print(f'Eval result:\n  Accuracy: {acc}\n  Mean returns: {returns}\n  Mean length: {lengths}')
        exit()

    env = DoorEnv(cfg.env, cfg.show_gui)

    if args.mode == 'bopt-gmm':
        conf_hash = conf_checksum(cfg)

        main_bopt_agent(env, cfg.bopt_agent, conf_hash, cfg.show_gui, args.wandb, args.run_prefix, args.model_dir)
    elif args.mode == 'eval-gmm':
        if cfg.bopt_agent.gmm.type not in GMM_TYPES:
            print(f'Unknown GMM type {cfg.bopt_agent.gmm.type}. Options are: {GMM_TYPES.keys()}')
            exit()

        gmm = GMM_TYPES[cfg.bopt_agent.gmm.type].load_model(cfg.bopt_agent.gmm.model)
        
        if args.wandb:
            logger = WBLogger('bopt-gmm', f'eval_{cfg.bopt_agent.gmm.model[:-4]}', False)
            logger.log_config({'type': cfg.bopt_agent.gmm.type, 
                               'model': cfg.bopt_agent.gmm.model})
        else: 
            logger = None

        acc, returns, lengths = evaluate(env, gmm,
                                         max_steps=600,
                                         num_episodes=100,
                                         show_force=cfg.show_gui,
                                         force_norm=cfg.bopt_agent.gmm.force_norm,
                                         logger=logger,
                                         const_gripper_cmd=cfg.bopt_agent.gripper_command)
        print(f'Eval result:\n  Accuracy: {acc}\n  Mean returns: {returns}\n  Mean length: {lengths}')
    
    # Pos GMM result: 52%
    # F-GMM result: 40%
