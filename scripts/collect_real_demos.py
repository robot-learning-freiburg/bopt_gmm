import hydra
import rospy

from argparse import ArgumentParser
from pathlib  import Path

from rl_franka.panda import Panda

from bopt_gmm.envs  import ENV_TYPES, \
                           RealDrawerEnv
from bopt_gmm.utils import save_demo_npz

if __name__ == '__main__':
    parser = ArgumentParser(description='Script to collect demos using hand guiding.')
    parser.add_argument('hy',      type=str, help='Hydra config for environment to use. Relative to root of "config" dir')
    parser.add_argument('out_dir', type=str, help='Directory to save demos to.')
    parser.add_argument('--overrides', default=[], type=str, nargs='*', help='Overrides for hydra config')
    args = parser.parse_args()

    if not Path(args.out_dir).exists():
        Path(args.out_dir).mkdir(parents=True)

    hydra.initialize(config_path="../config")
    cfg = hydra.compose(args.hy, overrides=args.overrides)

    env = ENV_TYPES[cfg.env.type](cfg.env, False) # type: RealDrawerEnv

    robot = env._robot  # type: Panda

    while not rospy.is_shutdown():
        robot.start_handguiding(0, 10)
        uinput = input('IS ROBOT FREE? (y/n): ')
        if uinput.lower() == 'y':
            robot.cm.activate_controller('position_joint_trajectory_controller')
            rospy.sleep(0.1)
            env.reset()
            print('ENVIRONMENT IS RESET')
            robot.cm.activate_controller('cartesian_impedance_controller')
            rospy.sleep(0.1)
            robot.start_handguiding(0, 10)
            print('HAND GUIDING ENABLED. GO!')

            trajectory = []
            while not rospy.is_shutdown() and not env.is_terminated()[0]:
                obs = env.observation()
                trajectory.append(obs)

                rospy.sleep(1 / 30)

            while True:
                uinput = input('---------------- DEMO ENDED ----------------\nSave demo? (y/n): ')
                if uinput.lower() == 'y':
                    save_demo_npz(observations=trajectory, save_dir=args.out_dir)
                    print('Saved demo')
                    break
                elif uinput.lower() == 'n':
                    print('Discarding demo.')
                    break

