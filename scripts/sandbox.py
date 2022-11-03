import hydra
import iai_bullet_sim as ibs
import numpy as np
import sys
import time

from tqdm import tqdm

# from drlfads.envs.iai_peg_env import PegEnv
# from drlfads.utils.path import pkg_path


if __name__ == '__main__':
    sim = ibs.BasicSimulator(60, real_time=True)
    sim.init('gui')

    ibs.add_search_path('drlfads/envs')
    ibs.add_search_path('drlfads/envs/data')

    robot = sim.load_urdf('package://bopt_gmm/robots/panda_hand.urdf', useFixedBase=True)
    eef   = robot.links['panda_hand_tcp']
    
    ft_sensor = robot.get_ft_sensor('panda_hand_joint')

    q_r = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])
    robot.set_joint_positions(q_r, override_initial=True)

    controller = ibs.CartesianController(robot, eef)
    controller.reset()

    while True:
        controller.act(controller.goal)
        sim.update()
        wrench = ft_sensor.get()
        print(f'---\nLin: {wrench.linear}\nAng: {wrench.angular}')
    