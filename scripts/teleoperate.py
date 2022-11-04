import hydra
from bopt_gmm.sim_control import start_web_app
from bopt_gmm.envs        import PegEnv

from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser(description='Using hydra without losing control of your code.')
    parser.add_argument('hy',      type=str, help='Hydra config for environment to use. Relative to root of "config" dir')
    parser.add_argument('out_dir', type=str, help='Directory to save demos to.')
    parser.add_argument('--overrides', default=[], type=str, nargs='*', help='Overrides for hydra config')
    args = parser.parse_args()

    hydra.initialize(config_path="../config")
    cfg = hydra.compose(args.hy, overrides=args.overrides)

    env = PegEnv(cfg.env, True)
    start_web_app(env, args.out_dir)
