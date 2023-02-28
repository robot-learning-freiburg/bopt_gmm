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
from tqdm                       import tqdm

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

from bopt_gmm.baselines import SACGMMEnv, \
                               SACGMMEnvCallback
from bopt_gmm.common    import run_episode

np.object = object
np.bool   = bool

from stable_baselines3     import SAC
from stable_baselines3.sac import MlpPolicy
from tqdm import tqdm


class SACGMMExperimentHook(SACGMMEnvCallback):
    def __init__(self, logger, agent, data_dir=None, ep_offset=0, ep_end_eval=None):
        self.logger   = logger
        self.agent    = agent
        self.data_dir = data_dir
        self.episode_ended = False
        self.ep_count = ep_offset
        self._ep_end_eval = ep_end_eval
        self._bopt_steps  = 0

    def on_episode_start(self, *args):
        pass

    def on_episode_end(self, env, obs, env_steps, sac_steps):
        self.episode_ended = True
        self.ep_count     += 1

        if self._ep_end_eval is not None:
            self._ep_end_eval(self)

        if self.logger is not None:
            self.logger.log({'n episode': self.ep_count})            

    def on_reset(self, env_config):
        self.episode_ended = False

    def on_post_step(self, reward, steps):
        self._bopt_steps += 1
        self.logger.log({'bopt reward':     reward,
                         'bopt mean steps': steps,
                         BOPT_TIME_SCALE: self._bopt_steps})


def build_sacgmm(env, gmm_agent, gripper_command, sacgmm_config, logger):
    sacgmm_env = SACGMMEnv(env, gmm_agent, gripper_command, sacgmm_config)

    model = SAC('MlpPolicy', sacgmm_env, 
                learning_rate=sacgmm_config.learning_rate,
                buffer_size=int(sacgmm_config.replay_buffer_size),
                learning_starts=sacgmm_config.warm_start_steps,
                tau=sacgmm_config.tau,
                gamma=sacgmm_config.gamma,
                policy_kwargs=dict(
                    net_arch=dict(pi=sacgmm_config.actor.arch, qf=sacgmm_config.critic.arch),
                    n_critics=sacgmm_config.critic.num
                ))
    
    return model, sacgmm_env


def evaluate_sacgmm(env, sacgmm_agent : MlpPolicy, num_episodes, max_steps):
    returns = []

    successes = []

    for x in tqdm(range(num_episodes), desc='Evaluating model'):
        obs  = env.reset()
        done = False

        while not done:
            action = sacgmm_agent.predict(obs)[0]
            obs, reward, done, info = env.step(action)
        
        successes.append(int(info['success']))

    return np.mean(successes)


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

    gmm = GMM.load_model(boptgmm_config.gmm.model)
    gmm_agent = GMMOptAgent(gmm, boptgmm_config)
    model, sacgmm_env = build_sacgmm(env, gmm_agent, boptgmm_config.gripper_command, sacgmm_config, logger)

    max_ep_steps = (max_steps // sacgmm_config.sacgmm_steps)

    model.learn(cfg.sacgmm.warm_start_steps, reset_num_timesteps=False, progress_bar=True)
    
    sacgmm_env.reset()

    def ep_end_eval_cb(hook):
        if hook.episode_ended and deep_eval_length > 0 and hook.ep_count % ckpt_freq == 0:
            model.policy.set_training_mode(False)
            if data_dir is not None:
                model.save(f'{data_dir}/sacgmm_model_{hook.ep_count:02d}.npz')

            e_env = sacgmm_env.eval_copy()
            accuracy = evaluate_sacgmm(e_env, model, deep_eval_length, max_steps)
            logger.log({'bopt deep eval accuracy': accuracy})

    hook = SACGMMExperimentHook(logger, model, data_dir, ep_offset=sacgmm_env._ep_count, ep_end_eval=ep_end_eval_cb)

    # Add Logging callback post initialization
    sacgmm_env.add_callback(hook)
    total_eps = num_training_cycles * cfg.bopt_agent.early_tell
    while hook.ep_count < total_eps:
        model.policy.set_training_mode(True)
        model.learn(1, reset_num_timesteps=False)
        print(f'{hook.ep_count}/{total_eps}')

    if data_dir is not None:
        model.save(f'{data_dir}/model_final.npz')


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
    args = parser.parse_args()

    hydra.initialize(config_path="../config")
    cfg = hydra.compose(args.hy, overrides=args.overrides)
    env = ENV_TYPES[cfg.env.type](cfg.env, args.show_gui)

    if args.mode == 'sacgmm':
        conf_hash = conf_checksum(cfg)

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
        pass
    else:
        print(f'Unknown mode "{args.mode}"')
        exit(-1)
