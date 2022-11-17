import cv2
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
                             create_dpg_context, \
                             is_dpg_running, \
                             render_dpg_frame, \
                             MP4VideoLogger

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


def run_episode(env, agent, max_steps, post_step_hook=None):
    observation    = env.reset()
    episode_return = 0.0

    for step in range(max_steps):
        action = agent.predict(observation)
        # print(observation)
        post_observation, reward, done, info = env.step(action)
        episode_return += reward
        done = done or (step == max_steps - 1)

        if post_step_hook is not None:
            post_step_hook(step, env, agent, observation, post_observation, action, reward, done, info)

        observation = post_observation
        
        if done:
            break
    
    return episode_return, step, info


def post_step_hook_dispatcher(*hooks):
    def dispatcher(step, env, agent, obs, post_obs, action, reward, done, info):
        for h in hooks:
            h(step, env, agent, obs, post_obs, action, reward, done, info)
    return dispatcher


def post_step_hook_bopt(_, env, agent, obs, post_obs, action, reward, done, info):
    agent.step(obs, post_obs, action, reward, done)


def gen_video_logger_and_hook(dir_path, filename, image_size, frame_rate=30.0):
    logger = MP4VideoLogger(dir_path, filename, image_size)

    def video_hook(_, env, *args):
        logger.write_image(env.render()[:,:,::-1])
    
    return logger, video_hook


def gen_force_logger_and_hook():
    create_dpg_context()
    live_plot = LivePlot('Forces', {'force_x': 'Force X', 
                                    'force_y': 'Force Y',
                                    'force_z': 'Force Z'})
    
    def live_plot_hook(step, env, agent, obs, *args):
        live_plot.add_value('force_x', obs['force'][0])
        live_plot.add_value('force_y', obs['force'][1])
        live_plot.add_value('force_z', obs['force'][2])
        render_dpg_frame()

    return live_plot, live_plot_hook


def evaluate_agent(env, agent, num_episodes=100, max_steps=600, 
                   logger=None, video_dir=None, trajectory_dir=None, show_forces=False, verbose=0):
    episode_returns = []
    episode_lengths = []
    
    successful_episodes = 0

    if show_forces:
        live_plot, live_plot_hook = gen_force_logger_and_hook()
    else:
        live_plot_hook = None

    video_logger = None

    for ep in tqdm(range(num_episodes), desc='Evaluating Agent'):
        post_step_hooks = [] if live_plot_hook is None else [live_plot_hook]

        if video_dir is not None:
            video_logger, video_hook = gen_video_logger_and_hook(video_dir, f'eval_{ep:04d}', env.render_size[:2])
            post_step_hooks.append(video_hook)

        ep_return, step, info = run_episode(env, agent, max_steps, post_step_hook=post_step_hook_dispatcher(*post_step_hooks))
        
        episode_returns.append(ep_return)
        episode_lengths.append(step)        
        
        if info["success"]:
            successful_episodes += 1
            if verbose > 0:
                print(f'Number of successes: {successful_episodes}\nCurrent Accuracy: {successful_episodes / ep}')
            
            if video_logger is not None:
                video_logger.rename(f'eval_{ep:04d}_S')
        elif video_logger is not None:
            video_logger.rename(f'eval_{ep:04d}_F')

        accuracy = successful_episodes / (ep + 1)

        if logger is not None:
            logger.log({'accuracy': accuracy, 
                        'success': int(info['success']),
                        'reward': ep_return, 
                        'steps': step + 1})
    
    if verbose > 0:
        print(f'Evaluation result:\n  Accuracy: {acc}\n  Mean returns: {np.mean(returns)}\n  Mean length: {np.mean(lengths)}')

    return accuracy, episode_returns, episode_lengths


def bopt_training(env, agent, num_episodes, max_steps=600, checkpoint_freq=10, 
                  opt_model_dir=None, logger=None, video_dir=None, show_force=False):
    if logger is not None:
        logger.define_metric('bopt accuracy',   BOPT_TIME_SCALE)
        logger.define_metric('bopt reward',     BOPT_TIME_SCALE)
        logger.define_metric('bopt mean steps', BOPT_TIME_SCALE)

    if show_force:
        live_plot, live_plot_hook = gen_force_logger_and_hook()
    else:
        live_plot_hook = None

    for bopt_step in tqdm(range(num_episodes), desc="Training BOPT model"):
        ep_acc = RunAccumulator()
        
        # Save initial model and models every n-opts
        if opt_model_dir is not None:
            if agent.is_in_gp_stage() and bopt_step == 1:
                agent.base_model.save_model(f'{opt_model_dir}/gmm_base.npy')
            
            if bopt_step % checkpoint_freq == 0:
                agent.model.save_model(f'{opt_model_dir}/gmm_{bopt_step}.npy')

        # Setup post-step hooks  
        post_step_hooks = [post_step_hook_bopt]
        if live_plot_hook is not None:
            post_step_hooks.append(live_plot_hook)

        if video_dir is not None:
            video_logger, video_hook = gen_video_logger_and_hook(video_dir, f'bopt_{bopt_step:04d}', env.render_size[:2])
            post_step_hooks.append(video_hook)

        # Collecting data for next BOPT update
        while agent.get_bopt_step() == bopt_step:
            ep_return, step, info = run_episode(env, agent, max_steps, post_step_hook=post_step_hook_dispatcher(*post_step_hooks))
            ep_acc.log_run(step + 1, ep_return, info['success'])
        
        # Log results of execution from this step
        if logger is not None:
            bopt_mean_steps, bopt_reward, bopt_accuracy = ep_acc.get_stats()
            logger.log({'bopt mean steps': bopt_mean_steps,
                        'bopt reward'    : bopt_reward, 
                        'bopt accuracy'  : bopt_accuracy})

    # Save final model
    if opt_model_dir is not None:
        agent.model.save_model(f'{opt_model_dir}/gmm_final.npy')


ENV_TYPES = {'door': DoorEnv,
             'peg': PegEnv}

GMM_TYPES = {'position': GMMCart3D,
                'force': GMMCart3DForce,
               'torque': GMMCart3DTorque}


def load_gmm(gmm_config):
    return GMM_TYPES[gmm_config.type].load_model(gmm_config.model)


def main_bopt_agent(env, bopt_agent_config, conf_hash, show_force=True, wandb=False, log_prefix=None, data_dir=None, render_video=False):

    if bopt_agent_config.agent not in {'bopt-gmm', 'dbopt'}:
        raise Exception(f'Unkown agent type "{bopt_agent_config.agent}"')

    if data_dir is not None:
        data_dir = f'{data_dir}_{conf_hash}'
        p = Path(data_dir)
        if not p.exists():
            p.mkdir(parents=True)

        with open(f'{data_dir}/config.yaml', 'w') as cfile:
            cfile.write(OmegaConf.to_yaml(bopt_agent_config))
    
    model_dir = f'{data_dir}/models' if data_dir is not None else None
    video_dir = f'{data_dir}/video'  if render_video         else None

    if model_dir is not None and not Path(model_dir).exists():
        Path(model_dir).mkdir(parents=True)
    
    if video_dir is not None and not Path(video_dir).exists():
        Path(video_dir).mkdir(parents=True)

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
        elif bopt_agent_config.gmm_generator.type == 'em':
            em_config     = bopt_agent_config.gmm_generator
            gmm_generator = em_gmm_generator(GMMCart3DForce,
                                             em_config.n_priors,
                                             em_config.max_iter,
                                             em_config.tol,
                                             em_config.n_init)
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

    acc, return_mean, mean_ep_length = bopt_training(env, agent, num_episodes=200, max_steps=600, 
                                                     opt_model_dir=model_dir, logger=logger, 
                                                     video_dir=video_dir, show_force=show_force)
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
    parser.add_argument('--video', action='store_true', help='Write video.')
    parser.add_argument('--show-gui', action='store_true', help='Show interactive GUI')
    parser.add_argument('--data-dir', default=None, help='Directory to save models and data to. Will be created if non-existent')
    args = parser.parse_args()

    # Point hydra to the root of your config dir. Here it's hard-coded, but you can also
    # use "MY_MODULE.__path__" to localize it relative to your python package
    hydra.initialize(config_path="../config")
    cfg = hydra.compose(args.hy, overrides=args.overrides)
    env = ENV_TYPES[cfg.env.type](cfg.env, args.show_gui)

    if args.mode == 'bopt-gmm':
        conf_hash = conf_checksum(cfg)


        main_bopt_agent(env, cfg.bopt_agent, conf_hash, args.show_gui, 
                        args.wandb, args.run_prefix, 
                        args.data_dir, render_video=args.video)
    elif args.mode == 'eval-gmm':
        if cfg.bopt_agent.gmm.type not in GMM_TYPES:
            print(f'Unknown GMM type {cfg.bopt_agent.gmm.type}. Options are: {GMM_TYPES.keys()}')
            exit()

        gmm = GMM_TYPES[cfg.bopt_agent.gmm.type].load_model(cfg.bopt_agent.gmm.model)
        
        gmm_path = Path(cfg.bopt_agent.gmm.model)

        if args.wandb:
            logger = WBLogger('bopt-gmm', f'eval_{cfg.bopt_agent.gmm.model[:-4]}', False)
            logger.log_config({'type': cfg.bopt_agent.gmm.type, 
                               'model': cfg.bopt_agent.gmm.model})
        else: 
            logger = None

        if args.video and args.data_dir is not None:
            video_dir = f'{args.data_dir}_{gmm_path.name[:-4]}'
        else:
            video_dir = None

        agent = AgentWrapper(gmm, 
                             cfg.bopt_agent.gmm.force_norm, 
                             cfg.bopt_agent.gripper_command)

        acc, returns, lengths = evaluate_agent(env, agent,
                                               num_episodes=100,
                                               max_steps=600,
                                               logger=logger,
                                               video_dir=video_dir,
                                               show_forces=args.show_gui,
                                               verbose=1)
    
    # Pos GMM result: 52%
    # F-GMM result: 40%
