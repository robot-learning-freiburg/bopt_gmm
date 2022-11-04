import hydra
import iai_bullet_sim as ibs
import numpy as np
import sys
import time

from argparse      import ArgumentParser
from bopt_gmm.envs import PegEnv


if __name__ == '__main__':
    parser = ArgumentParser(description='Using hydra without losing control of your code.')
    parser.add_argument('hy',      type=str, help='Hydra config for environment to use. Relative to root of "config" dir')
    parser.add_argument('--overrides', default=[], type=str, nargs='*', help='Overrides for hydra config')
    args = parser.parse_args()

    hydra.initialize(config_path="../config")
    cfg = hydra.compose(args.hy, overrides=args.overrides)

    env = PegEnv(cfg.env, True)
    env.reset()

    # sim = ibs.BasicSimulator(60, real_time=True)
    # sim.init('gui')

    # robot = sim.load_urdf('package://bopt_gmm/robots/panda_hand.urdf', useFixedBase=True)
    # eef   = robot.links['panda_hand_tcp']
    
    # ft_sensor = robot.get_ft_sensor('panda_hand_joint')

    # q_r = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])
    # robot.set_joint_positions(q_r, override_initial=True)

    # controller = ibs.CartesianController(robot, eef)
    # controller.reset()

    while True:
        # controller.act(controller.goal)
        # sim.update()
        # wrench = ft_sensor.get()
        # print(f'---\nLin: {wrench.linear}\nAng: {wrench.angular}')
        _, _, done, _ = env.step({'motion': ibs.Vector3.zero().numpy(), 'gripper': -1})
        if done:
            env.reset()
        time.sleep(env.dt)
    