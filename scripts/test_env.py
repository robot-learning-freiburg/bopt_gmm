import hydra
import numpy as np
import rospy

from argparse import ArgumentParser

from bopt_gmm.envs import ENV_TYPES


if __name__ == '__main__':
    parser = ArgumentParser(description='Using hydra without losing control of your code.')
    parser.add_argument('hy', type=str, help='Hydra config to use. Relative to root of "config" dir')
    parser.add_argument('--overrides', default=[], type=str, nargs='*', help='Overrides for hydra config')
    args = parser.parse_args()

    # Point hydra to the root of your config dir. Here it's hard-coded, but you can also
    # use "MY_MODULE.__path__" to localize it relative to your python package
    hydra.initialize(config_path="../config")
    cfg = hydra.compose(args.hy, overrides=args.overrides)
    env = ENV_TYPES[cfg.env.type](cfg.env)

    env.reset()
    
    while not rospy.is_shutdown():
        rospy.sleep(0.1)
        print(env.observation())
        terminated, success = env.is_terminated()
        print(f'Is terminated: {terminated}\nIs success: {success}')
        if terminated:
            env.reset()
