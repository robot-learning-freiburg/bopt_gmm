import hydra
import numpy as np

from argparse import ArgumentParser
from pathlib  import Path


from bopt_gmm import common, \
                     envs,   \
                     gmm
                     

if __name__ == '__main__':
    parser = ArgumentParser(description='Runs a model until a given number of trajectories are collected.')
    parser.add_argument('hy', type=str, help='Hydra config to use. Relative to root of "config" dir')
    parser.add_argument('out', default=None, help='File to save trajectories to. Dirs will be created if non-existent')
    parser.add_argument('--samples', default=15, type=int, help='Number of successful trajectories to collect.')
    parser.add_argument('--show-gui', action='store_true', help='Show interactive GUI')
    parser.add_argument('--max-steps', default=600, type=int, help='Maximum number of steps to do per episode.')
    parser.add_argument('--overrides', default=[], type=str, nargs='*', help='Overrides for hydra config')
    args = parser.parse_args()
    
    hydra.initialize(config_path="../config")
    cfg = hydra.compose(args.hy, overrides=args.overrides)
    
    env = envs.ENV_TYPES[cfg.env.type](cfg.env, args.show_gui)

    model = gmm.GMM.load_model(cfg.bopt_agent.gmm.model)


    p = args.out if args.out.lower()[-4:] == '.npz' else f'{args.out}.npz'
    p = Path(p)

    if not p.parent.exists():
        p.parent.mkdir(parents=True)

    successful_samples = 0
    samples = 0
    trajectories = []

    def collect_point(_, env, agent, obs, post_obs, action, reward, done, info):
        trajectories[-1].append((obs, post_obs, action, reward, done))

    agent = common.AgentWrapper(model, cfg.bopt_agent.gripper_command)

    print('Generating trajectories...')
    while successful_samples < args.samples:
        trajectories.append([])

        reward, step, info = common.run_episode(env, agent, args.max_steps, collect_point)

        if info['success']:
            successful_samples += 1
            print(f'Generated {successful_samples}/{args.samples}')
        else:
            del trajectories[-1]

        samples += 1

    trajectories = np.asarray(trajectories, dtype=object)
    np.savez(p, trajectories)
    print(f'Model file: {p}\nOverall accuracy: {successful_samples / samples}')
