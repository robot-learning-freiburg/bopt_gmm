import rospy
import numpy as np

from argparse import ArgumentParser
from roebots  import ROSVisualizer

from bopt_gmm.gmm   import rollout, \
                           GMM
from bopt_gmm.utils import unpack_trajectories

if __name__ == '__main__':
    parser = ArgumentParser(description='RVIZ visualization of GMMs')
    parser.add_argument('gmm', help='GMM to visualize')
    parser.add_argument('trajectories', nargs='+', help='Trajectories to load as starting points and references.')
    parser.add_argument('--show-all', action='store_true', help='Show all rollouts at once or one-by-one.')
    parser.add_argument('--steps', default=300, type=int, help='Number of steps to rollout.')
    parser.add_argument('--action-freq', default=30, type=int, help='Prediction interval in Hz.')
    args = parser.parse_args()

    trajs = unpack_trajectories(args.trajectories, 
                                [np.load(t, allow_pickle=True) for t in args.trajectories])
    gmm = GMM.load_model(args.gmm)

    rospy.init_node('bopt_gmm_visualizer')

    vis = ROSVisualizer('vis')

    vis.begin_draw_cycle('gmm_path', 'demo', 'components')
    vis.draw_points('components', np.eye(4), 0.02, gmm.mu()[:, :3], r=1)
    vis.render('components')

    for _, _, _, t in trajs:
        gmm_traj = rollout(gmm, t[0, :gmm.n_dims // 2], args.steps, 1 / args.action_freq)
        vis.draw_strip('gmm_path', np.eye(4), 0.005, gmm_traj[:, :3], r=0.8, g=0.5)
        vis.draw_strip('demo', np.eye(4), 0.005, t[:, :3], r=0, b=0.9)
        vis.render()

        if not args.show_all:
            bla = input('Press enter to see next GMM')
            vis.begin_draw_cycle('gmm_path', 'demo')
    
    rospy.sleep(0.1) # For topics are stupid
