import cv2
import hydra
import numpy as np
import time
import yaml

import random

from argparse                   import ArgumentParser
from ConfigSpace                import ConfigurationSpace
from ConfigSpace                import UniformFloatHyperparameter
from dataclasses                import dataclass
from datetime                   import datetime
from math                       import inf as Infinity
from omegaconf                  import OmegaConf
from pathlib                    import Path
from skopt                      import gp_minimize
from tqdm                       import tqdm
from typing                     import Any

from bopt_gmm.bopt import BOPTGMMCollectAndOptAgent, \
                          BOPTGMMAgent,              \
                          BOPTAgentGMMConfig,        \
                          BOPTAgentGenGMMConfig,     \
                          OnlineGMMAgent,            \
                          OnlineGMMConfig,           \
                          BOPT_TIME_SCALE
import bopt_gmm.bopt.regularization as reg
                          
from bopt_gmm.gmm import GMM,             \
                         GMMCart3D,       \
                         GMMCart3DForce,  \
                         GMMCart3DTorque, \
                         load_gmm,        \
                         get_gmm_model,   \
                         utils as gmm_utils

from bopt_gmm import bopt, \
                     common, \
                     gmm

from bopt_gmm.baselines import LSTMPolicy, \
                               LSTMPolicyConfig, \
                               sac_gmm

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

from rl_tasks import ENV_TYPES



def gen_force_logger_and_hook(force_scale=1.0):
    create_dpg_context()
    live_plot = LivePlot('Forces', {'force_x': 'Force X', 
                                    'force_y': 'Force Y',
                                    'force_z': 'Force Z'})
    
    def live_plot_hook(step, env, agent, obs, *args):
        live_plot.add_value('force_x', obs['force'][0] * force_scale)
        live_plot.add_value('force_y', obs['force'][1] * force_scale)
        live_plot.add_value('force_z', obs['force'][2] * force_scale)
        render_dpg_frame()

    return live_plot, live_plot_hook

def evaluate_agent(env, agent, num_episodes=100, max_steps=600, 
                   logger=None, video_dir=None, trajectory_dir=None, 
                   show_forces=False, verbose=0, initial_conditions_path=None,
                   tqdm_desc='Evaluating Agent'):
    episode_returns = []
    episode_lengths = []
    
    successful_episodes = 0

    if show_forces:
        force_scale = agent.model._force_scale if hasattr(agent.model, '_force_scale') else 1.0
        live_plot, live_plot_hook = gen_force_logger_and_hook(force_scale)
    else:
        live_plot_hook = None

    if initial_conditions_path is not None:
        fields = env.config_space
        fields += ['episode', 'accuracy', 'steps', 'reward', 'success']

        ic_logger = CSVLogger(initial_conditions_path, fields)
    else:
        ic_logger = None

    video_logger = None

    visualizer = env._vis if hasattr(env, '_vis') else None

    if visualizer is not None:
        def live_plot_hook(step, env, agent, obs, *args):
            visualizer.begin_draw_cycle('gmm_stats')
            gmm_utils.draw_gmm_stats(visualizer, 'gmm_stats', agent.model, obs, frame=env._ref_frame)
            visualizer.render('gmm_stats')

        def reset_hook(env, observation):
            visualizer.begin_draw_cycle('gmm_rollout')
            rollout = gmm_utils.rollout(agent.model, np.hstack([observation[d] for d in agent.model.semantic_obs_dims()]), steps=600)
            visualizer.draw_strip('gmm_rollout', env._robot_T_ref, 0.005, rollout)
            visualizer.render('gmm_rollout')
    else:
        def reset_hook(*args):
            pass

    for ep in tqdm(range(num_episodes), desc=tqdm_desc):
        agent.reset()
        post_step_hooks = [] if live_plot_hook is None else [live_plot_hook]

        if video_dir is not None:
            video_logger, video_hook = common.gen_video_logger_and_hook(video_dir, f'eval_{ep:04d}', env.render_size[:2])
            post_step_hooks.append(video_hook)

        if visualizer is not None:
            time.sleep(0.3)
            gmm_utils.draw_gmm(visualizer, 'gmm_model', agent.model, dimensions=['position', 'position|velocity'], visual_scaling=0.2, frame=env._ref_frame)
            time.sleep(0.3)

        ep_return, step, info = common.run_episode(env, agent, max_steps, 
                                                   post_step_hook=common.post_step_hook_dispatcher(*post_step_hooks),
                                                   post_reset_hook=reset_hook)
        
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


@dataclass
class IncumbentStats:
    accuracy : float
    step     : int
    config   : Any


def bopt_training(env, agent, num_episodes, max_steps=600, checkpoint_freq=10, 
                  opt_model_dir=None, logger=None, video_dir=None, 
                  show_force=False, deep_eval_length=0, incumbent_eval=None,
                  replay_recorder=None):
    if logger is not None:
        logger.define_metric('bopt accuracy',   BOPT_TIME_SCALE)
        logger.define_metric('bopt reward',     BOPT_TIME_SCALE)
        logger.define_metric('bopt mean steps', BOPT_TIME_SCALE)
        logger.define_metric('n episode',       BOPT_TIME_SCALE)
        if deep_eval_length > 0:
            logger.define_metric('bopt deep eval accuracy', BOPT_TIME_SCALE)
        if incumbent_eval is not None:
            logger.define_metric('bopt deep tell', BOPT_TIME_SCALE)

    if opt_model_dir is not None:
        if replay_recorder is not None:
            rp_buffer_path = Path(opt_model_dir) / '../replay_buffer'
            rp_buffer_path.mkdir(exist_ok=True)

        ic_fields = env.config_space
        ic_fields += [BOPT_TIME_SCALE, 'substep', 'steps', 'success']

        config_fields = [BOPT_TIME_SCALE] + list(agent.config_space.keys()) + ['accuracy', 'episodes']

        # Fix location generation
        ic_logger     = CSVLogger(f'{opt_model_dir}/../bopt_initial_conditions.csv', ic_fields)
        config_logger = CSVLogger(f'{opt_model_dir}/../bopt_configs.csv', config_fields)
        incumbent_logger = CSVLogger(f'{opt_model_dir}/../bopt_incumbents.csv', config_fields)
    else:
        ic_logger        = None
        config_logger    = None
        incumbent_logger = None

    if show_force:
        live_plot, live_plot_hook = gen_force_logger_and_hook()
    else:
        live_plot_hook = None

    n_ep = 0
    n_incumbents = 0
    last_incumbent_config = -1
    
    base_performance : IncumbentStats = None
    best_performance : IncumbentStats = None
    last_performance : IncumbentStats = None
    very_best_performance : IncumbentStats = None


    for bopt_step in tqdm(range(num_episodes), desc="Training BOPT model"):
        ep_acc = common.RunAccumulator()
        
        incumbent_config = agent.get_incumbent_config()
        if last_incumbent_config != incumbent_config and incumbent_config is not None:
            n_incumbents += 1

        # Save initial model and models every n-opts
        if opt_model_dir is not None:
            if agent.is_in_gp_stage() and bopt_step == 0:
                agent.get_incumbent().save_model(f'{opt_model_dir}/gmm_base.npy')
            
            if incumbent_config != last_incumbent_config and n_incumbents % checkpoint_freq == 0:
                agent.get_incumbent().save_model(f'{opt_model_dir}/gmm_{bopt_step}.npy')
        
        # 
        if deep_eval_length > 0 and incumbent_config != last_incumbent_config and n_incumbents % checkpoint_freq == 0:
            eval_video_dir = f'{video_dir}/eval_{bopt_step}' if video_dir is not None else None
            if eval_video_dir is not None and not Path(eval_video_dir).exists():
                Path(eval_video_dir).mkdir(parents=True)

            # Fix location generation
            eval_ic_path = f'{opt_model_dir}/../eval_{bopt_step}_ic.csv' if opt_model_dir is not None else None

            eval_agent = common.AgentWrapper(agent.get_incumbent(),
                                             agent.config.gripper_command)

            e_acc, _, _ = evaluate_agent(env, eval_agent,
                                         num_episodes=deep_eval_length,
                                         max_steps=max_steps,
                                         video_dir=eval_video_dir,
                                         verbose=0,
                                         initial_conditions_path=eval_ic_path)
            
            logger.log({'bopt deep eval accuracy': e_acc})

            if incumbent_logger is not None and incumbent_config is not None:
                inc_log = {BOPT_TIME_SCALE: bopt_step,
                           'accuracy': e_acc,
                           'episodes': n_ep}
                inc_log.update(incumbent_config)
                incumbent_logger.log(inc_log)

        
        if incumbent_eval is not None and incumbent_config != last_incumbent_config:
            eval_agent = common.AgentWrapper(agent.get_incumbent(),
                                             agent.config.gripper_command)

            e_acc, _, _ = evaluate_agent(env, eval_agent,
                                             num_episodes=incumbent_eval.episodes,
                                             max_steps=max_steps,
                                             video_dir=None,
                                             verbose=0,
                                             initial_conditions_path=None,
                                             tqdm_desc='Evaluating Incumbent for Tell')
            
            logger.log({'bopt deep tell': e_acc})
            
            n_ep  += incumbent_eval.episodes
            if last_incumbent_config != -1:
                if 1.0 - best_performance.accuracy * incumbent_eval.improvement_expectation >= e_acc:
                    if bopt_step - best_performance.step >= incumbent_eval.new_model_every:
                        print(f'Incumbent underperformed ({1.0 - best_performance.accuracy * incumbent_eval.improvement_expectation} >= {e_acc}) and too many steps have expired {bopt_step - best_performance.step}. Restarting...')
                        # New model is underperforming. Resetting process
                        agent.reset_optimizer()
                
                        best_performance = base_performance
                        last_performance = best_performance
                else:
                    print(f'New best performance is: {e_acc}')
                    best_performance = IncumbentStats(e_acc, bopt_step, agent.get_incumbent())
            else:
                print(f'Baseline performance is {e_acc}')
                base_performance = IncumbentStats(e_acc, 0, agent.get_incumbent())
                last_performance = base_performance
                best_performance = base_performance

            if bopt_step - last_performance.step > incumbent_eval.new_model_every:
                print(f'Generation of new incumbent took too long. Restarting...')
                last_performance = IncumbentStats(base_performance.accuracy, bopt_step, base_performance.config)
                # best_performance = last_performance
                agent.reset_optimizer()

        last_incumbent_config = incumbent_config

        # Setup post-step hooks  
        post_step_hooks = [common.post_step_hook_bopt]
        if live_plot_hook is not None:
            post_step_hooks.append(live_plot_hook)

        if replay_recorder is not None:
            post_step_hooks.append(replay_recorder.cb_post_step)

        if video_dir is not None:
            video_logger, video_hook = common.gen_video_logger_and_hook(video_dir, f'bopt_{bopt_step:04d}', env.render_size[:2])
            post_step_hooks.append(video_hook)

        bopt_config = agent.state.current_update.config

        # Collecting data for next BOPT update
        sub_ep_idx = 0

        if replay_recorder is not None:
            replay_recorder.tell_update(agent.state.current_update)

        while agent.get_bopt_step() == bopt_step:
            ep_return, step, info = common.run_episode(env, agent, max_steps, post_step_hook=common.post_step_hook_dispatcher(*post_step_hooks))
            ep_acc.log_run(step + 1, ep_return, info['success'])
            if ic_logger is not None:
                ic = info['initial_conditions']
                ic.update({BOPT_TIME_SCALE: bopt_step, 
                           'substep': sub_ep_idx, 
                           'steps' : step + 1, 
                           'success': info['success']})
                ic_logger.log(ic)
            sub_ep_idx += 1

            n_ep += 1
            if n_ep >= 500:
                break

            if sub_ep_idx > 40:
                n_ep = 500
                break

        if replay_recorder is not None:
            replay_recorder.save(rp_buffer_path)

        if config_logger is not None:
            _, _, bopt_accuracy = ep_acc.get_stats()
            config_dict = {BOPT_TIME_SCALE: bopt_step,
                           'accuracy': bopt_accuracy,
                           'episodes': n_ep}
            config_dict.update(bopt_config)
            config_logger.log(config_dict)

        if video_dir is not None:
            _, _, bopt_accuracy = ep_acc.get_stats()
            video_logger.rename(f'bopt_{bopt_step:04d}_{bopt_accuracy:1.3f}')

        # Log results of execution from this step
        if logger is not None:
            bopt_mean_steps, bopt_reward, bopt_accuracy = ep_acc.get_stats()
            logger.log({'bopt mean steps': bopt_mean_steps,
                        'bopt reward'    : bopt_reward, 
                        'bopt accuracy'  : bopt_accuracy,
                        'n episode'      : n_ep})

        if n_ep >= 500:
            break

    # Save final model
    if opt_model_dir is not None:
        best_model = best_performance.config if best_performance is not None else agent.get_incumbent()
        best_model.save_model(f'{opt_model_dir}/gmm_best.npy')

        eval_agent = common.AgentWrapper(best_model,
                                         agent.config.gripper_command)

        e_acc, _, _ = evaluate_agent(env, eval_agent,
                                          num_episodes=max(deep_eval_length, 20),
                                          max_steps=max_steps,
                                          video_dir=None,
                                          verbose=0,
                                          initial_conditions_path=None,
                                          tqdm_desc='Evaluating Final Incumbent')
        print(f'Final performance: {e_acc}')
        logger.log({'final performance': e_acc})


class ReplayBufferRecorder():
    def __init__(self, env, sacgmm_steps, sacgmm_obs=None) -> None:
        self._replay_buffer = sac_gmm.ReplayBuffer()

        if sacgmm_obs is None:
            self._sacgmm_obs = env.observation_space.keys()
        else:
            non_supported_obs = [o for o in sacgmm_obs if o not in env.observation_space]
            if len(non_supported_obs) > 0:
                raise RuntimeError('Observations specified for SACGMM replay buffer are not supported:\n{}Options are:\n{}'.format(
                    ', '.join(non_supported_obs), ','.join(env.observation_space.keys())
                ))
            self._sacgmm_obs = sacgmm_obs
        
        self._episode_ends = []
        self._sacgmm_steps = sacgmm_steps
        self.reset()

    def reset_counter(self):
        self._current_step = 0
        self._current_pre_obs = None
        self._reward = 0

    def reset(self):
        self.reset_counter()
        self._current_update  = None

    def tell_update(self, update):
        self._current_update = update

    def cb_post_step(self, step, env, agent, obs, post_obs, action, reward, done, info):
        self._reward       += reward
        self._current_step += 1

        if self._current_pre_obs is None:
            self._current_pre_obs = {k: v for k, v in obs.items() if k in self._sacgmm_obs}
        elif self._current_step == self._sacgmm_steps or done:
            self._replay_buffer.add_transition(self._current_pre_obs,
                                               self._current_update,
                                               {k: v for k, v in post_obs.items() if k in self._sacgmm_obs},
                                               self._reward,
                                               done)
            if done:
                self._episode_ends.append(len(self._replay_buffer))
            self.reset_counter()

    def save(self, path : Path):
        self._replay_buffer.save(path)
        with open(path / 'rp_meta.yaml', 'w') as f:
            yaml.dump({'num_episodes': len(self._episode_ends),
                       'episode_ends': self._episode_ends}, f)


def main_bopt_agent(env, bopt_agent_config, conf_hash, 
                    show_force=True, wandb=False, log_prefix=None, 
                    data_dir=None, render_video=False, deep_eval_length=0,
                    trajectories=None, ckpt_freq=10, sacgmm_steps=None, sacgmm_obs=None):

    if bopt_agent_config.agent not in {'bopt-gmm', 'dbopt', 'online'}:
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

    if bopt_agent_config.agent in {'bopt-gmm', 'online'}:
        n_priors   = base_gmm.n_priors            if bopt_agent_config.agent == 'online' else None
        gmm_type   = type(base_gmm)               if bopt_agent_config.agent == 'online' else None
        modalities = base_gmm.semantic_obs_dims() if bopt_agent_config.agent == 'online' else None

        if bopt_agent_config.gmm_generator.type == 'seds':
            seds_config = bopt_agent_config.gmm_generator

            gmm_generator = seds_gmm_generator(seds_config.seds_path,
                                               GMMCart3DForce if gmm_type is None else gmm_type,
                                               seds_config.n_priors if n_priors is None else n_priors,
                                               seds_config.objective,
                                               seds_config.tol_cutting,
                                               seds_config.max_iter)
        elif bopt_agent_config.gmm_generator.type == 'em':
            em_config     = bopt_agent_config.gmm_generator
            gmm_generator = em_gmm_generator(get_gmm_model(em_config.model) if gmm_type is None else gmm_type,
                                             em_config.n_priors if n_priors is None else n_priors,
                                             em_config.max_iter,
                                             em_config.tol,
                                             em_config.n_init,
                                             em_config.modalities if modalities is None else modalities,
                                             em_config.normalize)
        else:
            raise Exception(f'Unknown GMM generator "{bopt_agent_config.gmm_generator}"')

        if bopt_agent_config.agent == 'bopt-gmm':
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
                                           budget_min=bopt_agent_config.budget_min,
                                           budget_max=bopt_agent_config.budget_max,
                                           n_trials=bopt_agent_config.n_trials,
                                           f_gen_gmm=gmm_generator,
                                           debug_data_path=f'{model_dir}/{bopt_agent_config.debug_data_path}',
                                           debug_gmm_path=f'{model_dir}/{bopt_agent_config.debug_gmm_path}')

            agent  = BOPTGMMCollectAndOptAgent(base_gmm, config, logger=logger)
        else:
            data_paths = list(Path(bopt_agent_config.data_path).parent.glob(Path(bopt_agent_config.data_path).name))
            trajectories = unpack_trajectories(data_paths, [np.load(t, allow_pickle=True) for t in data_paths],
                                               base_gmm.semantic_obs_dims())

            transition_trajs = []
            for _, dim_names, groups, data in trajectories:
                group_names = [dim_names[g[0]].split('_')[0] for g in groups]
                transitions = []

                for x, row in enumerate(data):
                    obs  = {gn: np.take(row, g) for g, gn in zip(groups, group_names)}
                    done = x == data.shape[0] - 1
                    transitions.append((obs, None, done * 100, done, None))
                
                transition_trajs.append(transitions)

            config = OnlineGMMConfig(original_data=transition_trajs,
                                     delta_t=env.dt,
                                     f_gen_gmm=gmm_generator,
                                     debug_gmm_path=f'{model_dir}/{bopt_agent_config.debug_gmm_path}',
                                     gripper_command=bopt_agent_config.gripper_command)
            agent = OnlineGMMAgent(base_gmm, config, logger=logger)
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
                                    base_accuracy=bopt_agent_config.base_accuracy,
                                    opt_dims=bopt_agent_config.opt_dims,
                                    max_training_steps=bopt_agent_config.num_training_cycles,
                                    budget_min=bopt_agent_config.budget_min,
                                    budget_max=bopt_agent_config.budget_max,
                                    n_trials=bopt_agent_config.n_trials)

        if bopt_agent_config.gmm.type in {'force', 'torque'}:
            # Not used anymore as observation processing is now done by the GMM
            def obs_transform_force_norm(obs):
                if 'force' in obs:
                    obs['force'] = obs['force'] * bopt_agent_config.gmm.force_norm
                return obs

            agent = BOPTGMMAgent(base_gmm, config, logger=logger)
        else:
            agent = BOPTGMMAgent(base_gmm, config, logger=logger)

    rp_recorder = ReplayBufferRecorder(env, sacgmm_steps, sacgmm_obs) if sacgmm_steps is not None and model_dir is not None else None

    bopt_training(env, agent,
                  num_episodes=bopt_agent_config.num_training_cycles, 
                  max_steps=bopt_agent_config.num_episode_steps, 
                  opt_model_dir=model_dir, logger=logger, 
                  video_dir=video_dir, show_force=show_force,
                  checkpoint_freq=ckpt_freq,
                  deep_eval_length=deep_eval_length,
                  incumbent_eval=bopt_agent_config.incumbent_eval,
                  replay_recorder=rp_recorder)
    # print(f'Accuracy: {acc} Mean return: {return_mean} Mean ep length: {mean_ep_length}')
    # bopt_res = agent.state.gp_optimizer.get_result()
    # print(f'F means: {bopt_res.x}\nReward: {bopt_res.fun}')



if __name__ == '__main__':
    parser = ArgumentParser(description='Using hydra without losing control of your code.')
    parser.add_argument('hy', type=str, help='Hydra config to use. Relative to root of "config" dir')
    parser.add_argument('--overrides', default=[], type=str, nargs='*', help='Overrides for hydra config')
    parser.add_argument('--trajectories', default=[], nargs='*', help='Trajectories to use for regularization')
    parser.add_argument('--mode', default='bopt-gmm', help='Modes to run the program in.', choices=['bopt-gmm', 'eval-gmm', 'eval-lstm'])
    parser.add_argument('--run-prefix', default=None, help='Prefix for the generated run-id for the logger')
    parser.add_argument('--wandb', action='store_true', help='Enable W&B logging.')
    parser.add_argument('--video', action='store_true', help='Write video.')
    parser.add_argument('--show-gui', action='store_true', help='Show interactive GUI')
    parser.add_argument('--deep-eval', default=0, type=int, help='Number of deep evaluation episodes to perform during bopt training.')
    parser.add_argument('--data-dir', default=None, help='Directory to save models and data to. Will be created if non-existent')
    parser.add_argument('--ckpt-freq', default=10, type=int, help='Frequency at which to save and evaluate models')
    parser.add_argument('--eval-out', default=None, help='File to write results of evaluation to. Will write in append mode.')
    parser.add_argument('--bc-inputs', default=['position'], nargs='+', help='Observations to feed to BC policy.')
    parser.add_argument('--seed', default=None, type=int, help='Fixes the seed of numpy and random.')
    parser.add_argument('--sacgmm-steps', default=None, type=int, help='Size of a sacgmm step. If set will trigger replay buffer generation.')
    parser.add_argument('--sacgmm-obs', default=None, nargs='*', type=set, help='Filter for observations recorded in sacgmm replay buffer.')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)


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
                        deep_eval_length=args.deep_eval, trajectories=trajs,
                        ckpt_freq=args.ckpt_freq,
                        sacgmm_steps=args.sacgmm_steps,
                        sacgmm_obs=args.sacgmm_obs)

    elif args.mode == 'eval-gmm':
        gmm_path = Path(cfg.bopt_agent.gmm.model)
        gmm = GMM.load_model(cfg.bopt_agent.gmm.model)
        
        if 'noise' in cfg.env:
            print(f'Noise: {cfg.env.noise}')

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

        args.deep_eval = 10 if args.deep_eval == 0 else args.deep_eval

        acc, returns, lengths = evaluate_agent(env, agent,
                                               num_episodes=args.deep_eval,
                                               max_steps=cfg.bopt_agent.num_episode_steps,
                                               logger=logger,
                                               video_dir=video_dir,
                                               show_forces=False, # args.show_gui and not 'real' in cfg.env.type,
                                               verbose=1)
    
        # Restore the position of a real robot
        env.reset()

        if args.eval_out is not None:
            with open(args.eval_out, 'w') as f:
                f.write('model,env,noise,accuracy,date\n')
                f.write(f'{cfg.bopt_agent.gmm.model},{cfg.env.type},{cfg.env.noise.position.variance},{acc},{datetime.now()}\n')
    
    elif args.mode == 'eval-lstm':
        policy_path = Path(cfg.bopt_agent.gmm.model)
        policy = LSTMPolicy.load_model(cfg.bopt_agent.gmm.model)
        

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
            video_dir = f'{args.data_dir}_{policy_path.name[:-4]}'
        else:
            video_dir = None

        agent = common.TorchAgentWrapper(policy, 
                                        cfg.bopt_agent.gripper_command, 
                                        observations=args.bc_inputs)

        args.deep_eval = 10 if args.deep_eval == 0 else args.deep_eval

        acc, returns, lengths = evaluate_agent(env, agent,
                                               num_episodes=args.deep_eval,
                                               max_steps=cfg.bopt_agent.num_episode_steps,
                                               logger=logger,
                                               video_dir=video_dir,
                                               show_forces=args.show_gui,
                                               verbose=1)
    
        if args.eval_out is not None:
            with open(args.eval_out, 'w') as f:
                f.write('model,env,noise,accuracy,date\n')
                f.write(f'{cfg.bopt_agent.gmm.model},{cfg.env.type},{cfg.env.noise.position.variance},{acc},{datetime.now()}\n')


    # Pos GMM result: 52%
    # F-GMM result: 40%
