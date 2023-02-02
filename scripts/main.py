import cv2
import hydra
import numpy as np

from argparse                   import ArgumentParser
from ConfigSpace                import ConfigurationSpace
from ConfigSpace                import UniformFloatHyperparameter
from dataclasses                import dataclass
from datetime                   import datetime
from math                       import inf as Infinity
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
import bopt_gmm.bopt.regularization as reg
                          
from bopt_gmm.gmm import GMM,             \
                         GMMCart3D,       \
                         GMMCart3DForce,  \
                         GMMCart3DTorque, \
                         GMM_TYPES,       \
                         load_gmm

from bopt_gmm import bopt, \
                     common, \
                     gmm

from bopt_gmm.gmm.generation import seds_gmm_generator, \
                                    em_gmm_generator

from bopt_gmm.utils   import conf_checksum, \
                             unpack_trajectories
from bopt_gmm.logging import WBLogger, \
                             BlankLogger, \
                             LivePlot, \
                             create_dpg_context, \
                             is_dpg_running, \
                             render_dpg_frame, \
                             MP4VideoLogger, \
                             CSVLogger

from bopt_gmm.envs import PegEnv,   \
                          DoorEnv,  \
                          ENV_TYPES


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
                   logger=None, video_dir=None, trajectory_dir=None, 
                   show_forces=False, verbose=0, initial_conditions_path=None):
    episode_returns = []
    episode_lengths = []
    
    successful_episodes = 0

    if show_forces:
        live_plot, live_plot_hook = gen_force_logger_and_hook()
    else:
        live_plot_hook = None

    if initial_conditions_path is not None:
        fields = env.config_space
        fields += ['episode', 'accuracy', 'steps', 'reward', 'success']

        ic_logger = CSVLogger(initial_conditions_path, fields)
    else:
        ic_logger = None

    video_logger = None

    for ep in tqdm(range(num_episodes), desc='Evaluating Agent'):
        post_step_hooks = [] if live_plot_hook is None else [live_plot_hook]

        if video_dir is not None:
            video_logger, video_hook = common.gen_video_logger_and_hook(video_dir, f'eval_{ep:04d}', env.render_size[:2])
            post_step_hooks.append(video_hook)

        ep_return, step, info = common.run_episode(env, agent, max_steps, post_step_hook=common.post_step_hook_dispatcher(*post_step_hooks))
        
        episode_returns.append(ep_return)
        episode_lengths.append(step)        
        
        if info["success"]:
            successful_episodes += 1
            if verbose > 0:
                print(f'Number of successes: {successful_episodes}\nCurrent Accuracy: {successful_episodes / (ep + 1)}')
            
            if video_logger is not None:
                video_logger.rename(f'eval_{ep:04d}_S')
        elif video_logger is not None:
            video_logger.rename(f'eval_{ep:04d}_F')

        accuracy = successful_episodes / (ep + 1)

        stats = {'accuracy': accuracy, 
                 'success': int(info['success']),
                 'reward': ep_return, 
                 'steps': step + 1}

        if logger is not None:
            logger.log(stats)
        
        if ic_logger is not None:
            ic = info['initial_conditions']
            ic.update(stats)
            ic['episode'] = ep
            ic_logger.log(ic)

    if verbose > 0:
        print(f'Evaluation result:\n  Accuracy: {accuracy}\n  Mean returns: {np.mean(episode_returns)}\n  Mean length: {np.mean(episode_lengths)}')

    return accuracy, episode_returns, episode_lengths


def bopt_training(env, agent, num_episodes, max_steps=600, checkpoint_freq=10, 
                  opt_model_dir=None, logger=None, video_dir=None, 
                  show_force=False, deep_eval_length=0):
    if logger is not None:
        logger.define_metric('bopt accuracy',   BOPT_TIME_SCALE)
        logger.define_metric('bopt reward',     BOPT_TIME_SCALE)
        logger.define_metric('bopt mean steps', BOPT_TIME_SCALE)
        if deep_eval_length > 0:
            logger.define_metric('bopt deep eval accuracy', BOPT_TIME_SCALE)

    if opt_model_dir is not None:
        fields = env.config_space
        fields += ['bopt_step', 'substep', 'steps', 'success']

        # Fix location generation
        ic_logger = CSVLogger(f'{opt_model_dir}/../bopt_initial_conditions.csv', fields)
    else:
        ic_logger = None

    if show_force:
        live_plot, live_plot_hook = gen_force_logger_and_hook()
    else:
        live_plot_hook = None

    for bopt_step in tqdm(range(num_episodes), desc="Training BOPT model"):
        ep_acc = common.RunAccumulator()
        
        # Save initial model and models every n-opts
        if opt_model_dir is not None:
            if agent.is_in_gp_stage() and bopt_step == 1:
                agent.base_model.save_model(f'{opt_model_dir}/gmm_base.npy')
            
            if bopt_step % checkpoint_freq == 0:
                agent.model.save_model(f'{opt_model_dir}/gmm_{bopt_step}.npy')
        
        # 
        if bopt_step % checkpoint_freq == 0 and deep_eval_length > 0:
            eval_video_dir = f'{video_dir}/eval_{bopt_step}' if video_dir is not None else None
            if eval_video_dir is not None and not Path(eval_video_dir).exists():
                Path(eval_video_dir).mkdir(parents=True)

            # Fix location generation
            eval_ic_path = f'{opt_model_dir}/../eval_{bopt_step}_ic.csv' if opt_model_dir is not None else None

            eval_agent = common.AgentWrapper(agent.model,
                                             agent.config.gripper_command)

            e_acc, _, _ = evaluate_agent(env, eval_agent,
                                         num_episodes=deep_eval_length,
                                         max_steps=max_steps,
                                         video_dir=eval_video_dir,
                                         verbose=0,
                                         initial_conditions_path=eval_ic_path)
            
            logger.log({'bopt deep eval accuracy': e_acc})

        # Setup post-step hooks  
        post_step_hooks = [common.post_step_hook_bopt]
        if live_plot_hook is not None:
            post_step_hooks.append(live_plot_hook)

        if video_dir is not None:
            video_logger, video_hook = common.gen_video_logger_and_hook(video_dir, f'bopt_{bopt_step:04d}', env.render_size[:2])
            post_step_hooks.append(video_hook)

        # Collecting data for next BOPT update
        sub_ep_idx = 0
        while agent.get_bopt_step() == bopt_step:
            ep_return, step, info = common.run_episode(env, agent, max_steps, post_step_hook=common.post_step_hook_dispatcher(*post_step_hooks))
            ep_acc.log_run(step + 1, ep_return, info['success'])
            if ic_logger is not None:
                ic = info['initial_conditions']
                ic.update({'bopt_step': bopt_step, 
                           'substep': sub_ep_idx, 
                           'steps' : step + 1, 
                           'success': info['success']})
                ic_logger.log(ic)
            sub_ep_idx += 1

        if video_dir is not None:
            _, _, bopt_accuracy = ep_acc.get_stats()
            video_logger.rename(f'bopt_{bopt_step:04d}_{bopt_accuracy:1.3f}')

        # Log results of execution from this step
        if logger is not None:
            bopt_mean_steps, bopt_reward, bopt_accuracy = ep_acc.get_stats()
            logger.log({'bopt mean steps': bopt_mean_steps,
                        'bopt reward'    : bopt_reward, 
                        'bopt accuracy'  : bopt_accuracy})

    # Save final model
    if opt_model_dir is not None:
        agent.model.save_model(f'{opt_model_dir}/gmm_final.npy')


def bopt_regularized_training(env, agent, reg_data, regularizer, 
                              num_episodes, min_reg_value=0,
                              max_steps=600, checkpoint_freq=10, 
                              opt_model_dir=None, logger=None, video_dir=None, 
                              show_force=False, deep_eval_length=0):
    if logger is not None:
        logger.define_metric('bopt accuracy',   BOPT_TIME_SCALE)
        logger.define_metric('bopt reward',     BOPT_TIME_SCALE)
        logger.define_metric('bopt mean steps', BOPT_TIME_SCALE)
        logger.define_metric('n episode',       BOPT_TIME_SCALE)
        logger.define_metric('bopt reg value',  BOPT_TIME_SCALE)
        logger.define_metric('bopt ep run',     BOPT_TIME_SCALE)
        if deep_eval_length > 0:
            logger.define_metric('bopt deep eval accuracy', BOPT_TIME_SCALE)

    if opt_model_dir is not None:
        fields = env.config_space
        fields += ['bopt_step', 'substep', 'steps', 'success']

        # Fix location generation
        ic_logger = CSVLogger(f'{opt_model_dir}/../bopt_initial_conditions.csv', fields)
    else:
        ic_logger = None

    if show_force:
        live_plot, live_plot_hook = gen_force_logger_and_hook()
    else:
        live_plot_hook = None

    def post_step_hook_bopt_reg(_, env, agent, obs, post_obs, action, reward, done, info):
        reg_val = regularizer(agent.model, agent.base_model, reg_data) if done else 0
        common.post_step_hook_bopt(_, env, agent, obs, post_obs, action, reward + reg_val, done, info)

    for n_ep in tqdm(range(num_episodes), desc="Training regularized BOPT model"):
        ep_acc = common.RunAccumulator()
        
        # Save initial model and models every n-opts
        if opt_model_dir is not None:
            if agent.is_in_gp_stage() and agent.get_bopt_step() == 1:
                agent.base_model.save_model(f'{opt_model_dir}/gmm_base.npy')
            
            if n_ep % checkpoint_freq == 0:
                agent.model.save_model(f'{opt_model_dir}/gmm_{n_ep}.npy')
        
        # 
        if n_ep > 0 and n_ep % checkpoint_freq == 0 and deep_eval_length > 0:
            # Fix location generation
            eval_ic_path = f'{opt_model_dir}/../eval_{n_ep}_ic.csv' if opt_model_dir is not None else None

            eval_agent = common.AgentWrapper(agent.model, 
                                      agent.state.obs_transform, 
                                      agent.config.gripper_command)

            e_acc, _, _ = evaluate_agent(env, eval_agent,
                                         num_episodes=deep_eval_length,
                                         max_steps=max_steps,
                                         verbose=0,
                                         initial_conditions_path=eval_ic_path)
            
            logger.log({'bopt deep eval accuracy': e_acc})

        for tries in range(1000):
            reg_val = regularizer(agent.model, agent.base_model, reg_data)
            logger.log({'bopt reg value' : reg_val})
            if min_reg_value <= reg_val:
                break
            
            # Log reg value as reward when it is used to update the model
            logger.log({'bopt reward' : reg_val,
                        'bopt ep run' : 0})
            agent.step_optimizer(reg_val)
        else:
            raise Exception(f'Failed to generate a feasible update complying with regularization in {tries} attempts')

        # Setup post-step hooks  
        post_step_hooks = [post_step_hook_bopt_reg]

        # Collecting data for next BOPT update
        ep_return, step, info = common.run_episode(env, agent, max_steps, post_step_hook=common.post_step_hook_dispatcher(*post_step_hooks))
        ep_acc.log_run(step + 1, ep_return, info['success'])
        if ic_logger is not None:
            ic = info['initial_conditions']
            ic.update({'bopt_step': agent.get_bopt_step(), 
                       'substep'  : 1, 
                       'steps'    : step + 1, 
                       'success'  : info['success']})
            ic_logger.log(ic)

        # if video_dir is not None:
        #     _, _, bopt_accuracy = ep_acc.get_stats()
        #     video_logger.rename(f'bopt_{bopt_step:04d}_{bopt_accuracy:1.3f}')

        # Log results of execution from this step
        if logger is not None:
            bopt_mean_steps, bopt_reward, bopt_accuracy = ep_acc.get_stats()
            logger.log({'bopt mean steps': bopt_mean_steps,
                        'bopt reward'    : bopt_reward, 
                        'bopt accuracy'  : bopt_accuracy,
                        'n episode'      : n_ep,
                        'bopt ep run'    : 1})

    # Save final model
    if opt_model_dir is not None:
        agent.model.save_model(f'{opt_model_dir}/gmm_final.npy')



def main_bopt_agent(env, bopt_agent_config, conf_hash, 
                    show_force=True, wandb=False, log_prefix=None, 
                    data_dir=None, render_video=False, deep_eval_length=0, trajectories=None):

    if bopt_agent_config.agent not in {'bopt-gmm', 'dbopt'}:
        raise Exception(f'Unkown agent type "{bopt_agent_config.agent}"')
    
    model_dir = f'{data_dir}/models' if data_dir is not None else None
    video_dir = f'{data_dir}/video'  if render_video         else None

    if model_dir is not None and not Path(model_dir).exists():
        Path(model_dir).mkdir(parents=True)
    
    if video_dir is not None and not Path(video_dir).exists():
        Path(video_dir).mkdir(parents=True)

    base_gmm = load_gmm(bopt_agent_config.gmm)

    run_id = f'{bopt_agent_config.agent}_{conf_hash}'
    run_id = f'{log_prefix}_{run_id}' if log_prefix is not None else run_id

    logger = WBLogger('bopt-gmm', run_id, True) if wandb else BlankLogger()
    logger.log_config(bopt_agent_config)

    if 'var_adjustment' in bopt_agent_config.gmm and bopt_agent_config.gmm.var_adjustment != 0:
        base_gmm = base_gmm.update_gaussian(sigma=np.stack([np.eye(base_gmm.n_dims) * bopt_agent_config.gmm.var_adjustment]*base_gmm.n_priors, axis=0))

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
                                       sigma_range=bopt_agent_config.sigma_range,
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
                                    sigma_range=bopt_agent_config.sigma_range,
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

        if bopt_agent_config.gmm.type in {'force', 'torque'}:
            # Not used anymore as observation processing is now done by the GMM
            def obs_transform_force_norm(obs):
                if 'force' in obs:
                    obs['force'] = obs['force'] * bopt_agent_config.gmm.force_norm
                return obs

            agent = BOPTGMMAgent(base_gmm, config, logger=logger)
        else:
            agent = BOPTGMMAgent(base_gmm, config, logger=logger)

    if 'regularizer' not in bopt_agent_config:
        bopt_training(env, agent, num_episodes=100, max_steps=600, 
                      opt_model_dir=model_dir, logger=logger, 
                      video_dir=video_dir, show_force=show_force, 
                      deep_eval_length=deep_eval_length)
    else:
        regularizer = reg.gen_regularizer(bopt_agent_config.regularizer)
        bopt_regularized_training(env, agent, np.vstack(trajectories), regularizer, 
                                  min_reg_value=bopt_agent_config.regularizer.min_val,
                                  num_episodes=100, max_steps=600, 
                                  opt_model_dir=model_dir, logger=logger, 
                                  video_dir=video_dir, show_force=show_force, 
                                  deep_eval_length=deep_eval_length)
    # print(f'Accuracy: {acc} Mean return: {return_mean} Mean ep length: {mean_ep_length}')
    # bopt_res = agent.state.gp_optimizer.get_result()
    # print(f'F means: {bopt_res.x}\nReward: {bopt_res.fun}')



if __name__ == '__main__':
    parser = ArgumentParser(description='Using hydra without losing control of your code.')
    parser.add_argument('hy', type=str, help='Hydra config to use. Relative to root of "config" dir')
    parser.add_argument('--overrides', default=[], type=str, nargs='*', help='Overrides for hydra config')
    parser.add_argument('--trajectories', default=[], nargs='*', help='Trajectories to use for regularization')
    parser.add_argument('--mode', default='bopt-gmm', help='Modes to run the program in.', choices=['bopt-gmm', 'eval-gmm', 'vis'])
    parser.add_argument('--run-prefix', default=None, help='Prefix for the generated run-id for the logger')
    parser.add_argument('--wandb', action='store_true', help='Enable W&B logging.')
    parser.add_argument('--video', action='store_true', help='Write video.')
    parser.add_argument('--show-gui', action='store_true', help='Show interactive GUI')
    parser.add_argument('--deep-eval', default=0, type=int, help='Number of deep evaluation episodes to perform during bopt training.')
    parser.add_argument('--data-dir', default=None, help='Directory to save models and data to. Will be created if non-existent')
    args = parser.parse_args()

    # Point hydra to the root of your config dir. Here it's hard-coded, but you can also
    # use "MY_MODULE.__path__" to localize it relative to your python package
    hydra.initialize(config_path="../config")
    cfg = hydra.compose(args.hy, overrides=args.overrides)
    env = ENV_TYPES[cfg.env.type](cfg.env, args.show_gui)

    if args.mode == 'bopt-gmm':
        conf_hash = conf_checksum(cfg)

        if args.data_dir is not None:
            args.data_dir = f'{args.data_dir}_{conf_hash}'
            p = Path(args.data_dir)
            if not p.exists():
                p.mkdir(parents=True)

            with open(f'{args.data_dir}/config.yaml', 'w') as cfile:
                cfile.write(OmegaConf.to_yaml(cfg))

        trajs = [d for _, _, _, d in unpack_trajectories(args.trajectories, 
                                                         [np.load(t, allow_pickle=True) for t in args.trajectories], 
                                                         ['position', 'force'])]

        main_bopt_agent(env, cfg.bopt_agent, conf_hash, args.show_gui, 
                        args.wandb, args.run_prefix, 
                        args.data_dir, render_video=args.video, 
                        deep_eval_length=args.deep_eval, trajectories=trajs)

    elif args.mode == 'eval-gmm':
        if cfg.bopt_agent.gmm.type not in GMM_TYPES:
            print(f'Unknown GMM type {cfg.bopt_agent.gmm.type}. Options are: {GMM_TYPES.keys()}')
            exit()

        gmm_path = Path(cfg.bopt_agent.gmm.model)
        gmm = GMM.load_model(cfg.bopt_agent.gmm.model)
        

        if args.wandb:
            run_name = f'eval_{cfg.bopt_agent.gmm.model[:-4]}'
            if args.run_prefix is not None:
                run_name = args.run_prefix
            logger = WBLogger('bopt-gmm', f'eval_{cfg.bopt_agent.gmm.model[:-4]}', False)
            logger.log_config({'type': cfg.bopt_agent.gmm.type, 
                               'model': cfg.bopt_agent.gmm.model,
                               'env': cfg.env})
        else: 
            logger = None

        if args.video and args.data_dir is not None:
            video_dir = f'{args.data_dir}_{gmm_path.name[:-4]}'
        else:
            video_dir = None

        agent = common.AgentWrapper(gmm, cfg.bopt_agent.gripper_command)

        acc, returns, lengths = evaluate_agent(env, agent,
                                               num_episodes=100,
                                               max_steps=600,
                                               logger=logger,
                                               video_dir=video_dir,
                                               show_forces=args.show_gui,
                                               verbose=1)
    
    # Pos GMM result: 52%
    # F-GMM result: 40%
