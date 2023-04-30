import cv2
import hydra
import numpy as np
import torch

from argparse      import ArgumentParser
from ConfigSpace   import ConfigurationSpace
from ConfigSpace   import UniformFloatHyperparameter
from dataclasses   import dataclass
from datetime      import datetime
from math          import inf as Infinity
from omegaconf     import OmegaConf
from pathlib       import Path
from skopt         import gp_minimize
from tqdm          import tqdm
from torch         import nn

from bopt_gmm.bopt import BOPTGMMCollectAndOptAgent, \
                          GMMOptAgent,               \
                          BOPTGMMAgent,              \
                          BOPTAgentGMMConfig,        \
                          BOPTAgentGenGMMConfig,     \
                          BOPT_TIME_SCALE
import bopt_gmm.bopt.regularization as reg
                          
from bopt_gmm.gmm import GMM,             \
                         GMMCart3D,       \
                         GMMCart3DForce,  \
                         GMMCart3DTorque, \
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

from bopt_gmm.baselines import sac_gmm
from bopt_gmm.common    import run_episode

np.object = object
np.bool   = bool

from stable_baselines3     import SAC
from stable_baselines3.sac import MlpPolicy
from tqdm import tqdm


def build_sacgmm(env, gmm_agent, gripper_command, sacgmm_config, logger, device='cuda'):
    sacgmm_env = sac_gmm.SACGMMEnv(env, gmm_agent, gripper_command, sacgmm_config, 
                                   obs_filter=set(sacgmm_config.observations))

    obs_dim    = int(np.product(sacgmm_env.observation_space.shape).item())
    action_dim = int(np.product(sacgmm_env.action_space.shape).item())

    actor = sac_gmm.DenseNetActor(obs_dim,
                                  sacgmm_env.action_space,
                                  sacgmm_config.actor.hidden_layers,
                                  sacgmm_config.actor.hidden_dim,
                                  sacgmm_config.actor.init_w,
                                  device)

    critic = sac_gmm.DenseNetCritic(obs_dim + action_dim,
                                    sacgmm_config.critic.hidden_layers,
                                    sacgmm_config.critic.hidden_dim,
                                    device)

    critic_target = sac_gmm.DenseNetCritic(obs_dim + action_dim,
                                           sacgmm_config.critic.hidden_layers,
                                           sacgmm_config.critic.hidden_dim,
                                           device)

    sac = sac_gmm.SACAgent(sacgmm_config,
                           actor,
                           critic, 
                           critic_target, 
                           sacgmm_env.action_space,
                           logger,
                           device)

    return sac, sacgmm_env


def evaluate_sacgmm(env : sac_gmm.SACGMMEnv, 
                    sacgmm_agent : sac_gmm.SACAgent, 
                    num_episodes, max_steps, ic_path=None):
    episode_returns = []
    episode_lengths = []
    
    successful_episodes = 0

    if ic_path is not None:
        fields = env.config_space
        fields += ['episode', 'accuracy', 'steps', 'reward', 'success']

        ic_logger = CSVLogger(ic_path, fields)
    else:
        ic_logger = None

    with torch.no_grad():
        for ep in tqdm(range(num_episodes), desc='Evaluating model'):
            obs  = env.reset()
            done = False

            initial_conditions = env.config_dict()

            while not done:
                action = sacgmm_agent.get_action(obs, 'deterministic')
                obs, reward, done, info = env.step(action)

                # Cap episode length
                done = done or env._n_env_steps >= max_steps
            
            episode_returns.append(reward)
            episode_lengths.append(env._n_env_steps)   

            if info["success"]:
                successful_episodes += 1

            stats = {'accuracy': float(info['success']), 
                     'success': int(info['success']),
                     'reward': reward, 
                     'steps': env._n_env_steps}

            if ic_logger is not None:
                ic = initial_conditions
                ic.update(stats)
                ic['episode'] = ep
                ic_logger.log(ic)

    return successful_episodes / ep, episode_returns, episode_lengths


def train_sacgmm(env, cfg, num_training_cycles, max_steps, 
                 wandb, data_dir, run_id, deep_eval_length=0, ckpt_freq=10):
    logger = WBLogger('bopt-gmm', run_id, True) if wandb else BlankLogger()
    logger.log_config(cfg)
    
    boptgmm_config = cfg.bopt_agent
    sacgmm_config  = cfg.sacgmm

    if logger is not None:
        logger.define_metric('bopt accuracy',   BOPT_TIME_SCALE)
        logger.define_metric('bopt reward',     BOPT_TIME_SCALE)
        logger.define_metric('bopt mean steps', BOPT_TIME_SCALE)
        logger.define_metric('n episode',       BOPT_TIME_SCALE)
        if deep_eval_length > 0:
            logger.define_metric('bopt deep eval accuracy', BOPT_TIME_SCALE)

    if data_dir is not None:
        fields = env.config_space
        fields += [BOPT_TIME_SCALE, 'steps', 'success']

        # Fix location generation
        ic_logger = CSVLogger(f'{data_dir}/sacgmm_initial_conditions.csv', fields)
    else:
        ic_logger = None

    gmm = GMM.load_model(boptgmm_config.gmm.model)
    gmm_agent = GMMOptAgent(gmm, boptgmm_config)
    sac_agent, sacgmm_env = build_sacgmm(env, gmm_agent, boptgmm_config.gripper_command, sacgmm_config, logger)

    if logger is not None:
        for m in sac_agent.metrics:        
            logger.define_metric(m, BOPT_TIME_SCALE)

    # We count all episodes, including the ones to fill replay
    ep_count = 0

    if True:
        # Fill replay buffer
        obs_prior = sacgmm_env.reset()
        for _ in tqdm(range(sac_agent.warm_start_steps), desc='Filling replay buffer...'):
            action = sac_agent.get_action(obs_prior, sacgmm_config.fill_strategy)
            obs_post, reward, done, info = sacgmm_env.step(action)
            
            # Cap episode length
            done = done or sacgmm_env._n_env_steps >= max_steps
            
            sac_agent.append_to_replay_buffer(obs_prior, action, obs_post, reward, done)

            if done:
                obs_prior = sacgmm_env.reset()
                ep_count += 1
            else:
                obs_prior = obs_post
    else:  # Fake filling of replay buffer for development purposes
        obs_prior = sacgmm_env.reset()
        action    = sac_agent.get_action(obs_prior, sacgmm_config.fill_strategy)
        
        for x in tqdm(range(sac_agent.warm_start_steps), desc='FAKE Filling replay buffer...'):
            done = x % 5 == 1
            sac_agent.append_to_replay_buffer(obs_prior, action, obs_prior, float(done) * 100, done)

    obs_prior = sacgmm_env.reset()
    if ic_logger is not None:
        ic = sacgmm_env.config_dict()

    # Add Logging callback post initialization
    total_eps = num_training_cycles * cfg.bopt_agent.early_tell
    opt_steps = 0

    pbar = tqdm(total=total_eps) # Initialise

    while ep_count < total_eps:
        action = sac_agent.get_action(obs_prior, 'stochastic')
        obs_post, reward, done, info = sacgmm_env.step(action)

        # Cap episode length
        done = done or sacgmm_env._n_env_steps >= max_steps

        sac_agent.append_to_replay_buffer(obs_prior, action, obs_post, reward, done)

        if logger is not None:
            logger.log({BOPT_TIME_SCALE: opt_steps})
        
        sac_agent.train_step()
        opt_steps += 1

        if done:
            ep_count += 1
            if logger is not None:
                logger.log({'bopt mean steps': sacgmm_env._n_env_steps,
                            'bopt reward'    : reward, 
                            'bopt accuracy'  : float(done),
                            'n episode'      : ep_count,
                            'bopt ep run'    : 1})
            
            if ic_logger is not None:
                ic.update({BOPT_TIME_SCALE: opt_steps, 
                           'substep': 0, 
                           'steps' : sacgmm_env._n_env_steps, 
                           'success': info['success']})

            if deep_eval_length > 0 and ep_count % ckpt_freq == 0:
                if data_dir is not None:
                    sac_agent.save(f'{data_dir}/sacgmm_model_{ep_count:02d}.npz')

                eval_ic_path = f'{data_dir}/eval_{ep_count}_ic.csv' if data_dir is not None else None

                accuracy, ep_returns, ep_lengths = evaluate_sacgmm(sacgmm_env, sac_agent, deep_eval_length, max_steps, ic_path=eval_ic_path)
                logger.log({'bopt deep eval accuracy': accuracy})

            obs_prior = sacgmm_env.reset()
            if ic_logger is not None:
                ic = sacgmm_env.config_dict()
        else:
            obs_prior = obs_post
        pbar.update(ep_count)
    pbar.close()

    if data_dir is not None:
        sac_agent.save(f'{data_dir}/model_final.npz')


if __name__ == '__main__':
    parser = ArgumentParser(description='Using hydra without losing control of your code.')
    parser.add_argument('hy', type=str, help='Hydra config to use. Relative to root of "config" dir')
    parser.add_argument('--overrides', default=[], type=str, nargs='*', help='Overrides for hydra config')
    parser.add_argument('--trajectories', default=[], nargs='*', help='Trajectories to use for regularization')
    parser.add_argument('--mode', default='sacgmm', help='Modes to run the program in.', choices=['sacgmm', 'eval', 'vis'])
    parser.add_argument('--run-prefix', default=None, help='Prefix for the generated run-id for the logger')
    parser.add_argument('--wandb', action='store_true', help='Enable W&B logging.')
    parser.add_argument('--video', action='store_true', help='Write video.')
    parser.add_argument('--show-gui', action='store_true', help='Show interactive GUI')
    parser.add_argument('--deep-eval', default=0, type=int, help='Number of deep evaluation episodes to perform during bopt training.')
    parser.add_argument('--data-dir', default=None, help='Directory to save models and data to. Will be created if non-existent')
    parser.add_argument('--ckpt-freq', default=10, type=int, help='Frequency at which to save and evaluate models')
    parser.add_argument('--eval-out', default=None, help='File to write results of evaluation to. Will write in append mode.')
    parser.add_argument('--sac-model', default=None, help='Path to model to resume training from or to evaluate.')
    args = parser.parse_args()

    hydra.initialize(config_path="../config")
    cfg = hydra.compose(args.hy, overrides=args.overrides)
    env = ENV_TYPES[cfg.env.type](cfg.env, args.show_gui)

    if cfg.sacgmm.observations is None:
        print(f'sacgmm.observations in hydra config cannot be null')
        exit(-1)

    if args.mode == 'sacgmm':
        conf_hash = conf_checksum(cfg)

        args.run_prefix = f'{args.run_prefix}_{conf_hash}'        

        if args.data_dir is not None:
            args.data_dir = f'{args.data_dir}_{conf_hash}'
            p = Path(args.data_dir)
            if not p.exists():
                p.mkdir(parents=True)

            with open(f'{args.data_dir}/config.yaml', 'w') as cfile:
                cfile.write(OmegaConf.to_yaml(cfg))

        train_sacgmm(env, cfg, 
                     num_training_cycles=cfg.bopt_agent.num_training_cycles, 
                     max_steps=cfg.bopt_agent.num_episode_steps, 
                     wandb=args.wandb, 
                     data_dir=args.data_dir, 
                     run_id=args.run_prefix,
                     deep_eval_length=args.deep_eval,
                     ckpt_freq=args.ckpt_freq)
    elif args.mode == 'eval':
        if args.sac_model is None:
            print('Need "--sac-model" argument for evaluation.')
            exit(-1)

        boptgmm_config = cfg.bopt_agent
        sacgmm_config  = cfg.sacgmm

        gmm = GMM.load_model(boptgmm_config.gmm.model)
        gmm_agent = GMMOptAgent(gmm, boptgmm_config)
        sac_agent, sacgmm_env = build_sacgmm(env, gmm_agent, boptgmm_config.gripper_command, sacgmm_config, None)

        sac_agent.load(args.sac_model)

        accuracy, ep_returns, ep_lengths = evaluate_sacgmm(sacgmm_env,
                                                           sac_agent, 
                                                           args.deep_eval,
                                                           max_steps=cfg.bopt_agent.num_episode_steps, 
                                                           ic_path=None)

        if args.eval_out is not None:
            with open(args.eval_out, 'w') as f:
                f.write('model,gmm,env,noise,accuracy,date\n')
                f.write(f'{args.sac_model},{cfg.bopt_agent.gmm.model},{cfg.env.type},{cfg.env.noise.position.variance},{accuracy},{datetime.now()}\n')

    else:
        print(f'Unknown mode "{args.mode}"')
        exit(-1)
