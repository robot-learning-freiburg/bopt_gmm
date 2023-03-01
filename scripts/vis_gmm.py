import rospy
import math
import numpy as np

from argparse import ArgumentParser
from roebots  import ROSVisualizer

from bopt_gmm.gmm   import rollout, \
                           GMM
from bopt_gmm.utils import unpack_trajectories


COLORS = [(1,   0,   0, 1),
          (0,   1,   0, 1),
          (0,   0,   1, 1),
          (1, 0.5,   0, 1),
          (1,   0, 0.5, 1),
          (0,   1, 0.5, 1)]


def real_quat_from_matrix(frame):
    tr = frame[0,0] + frame[1,1] + frame[2,2]

    if tr > 0:
        S = math.sqrt(tr+1.0) * 2 # S=4*qw
        qw = 0.25 * S
        qx = (frame[2,1] - frame[1,2]) / S
        qy = (frame[0,2] - frame[2,0]) / S
        qz = (frame[1,0] - frame[0,1]) / S
    elif frame[0,0] > frame[1,1] and frame[0,0] > frame[2,2]:
        S  = math.sqrt(1.0 + frame[0,0] - frame[1,1] - frame[2,2]) * 2 # S=4*qx
        qw = (frame[2,1] - frame[1,2]) / S
        qx = 0.25 * S
        qy = (frame[0,1] + frame[1,0]) / S
        qz = (frame[0,2] + frame[2,0]) / S
    elif frame[1,1] > frame[2,2]:
        S  = math.sqrt(1.0 + frame[1,1] - frame[0,0] - frame[2,2]) * 2 # S=4*qy
        qw = (frame[0,2] - frame[2,0]) / S
        qx = (frame[0,1] + frame[1,0]) / S
        qy = 0.25 * S
        qz = (frame[1,2] + frame[2,1]) / S
    else:
        S  = math.sqrt(1.0 + frame[2,2] - frame[0,0] - frame[1,1]) * 2 # S=4*qz
        qw = (frame[1,0] - frame[0,1]) / S
        qx = (frame[0,2] + frame[2,0]) / S
        qy = (frame[1,2] + frame[2,1]) / S
        qz = 0.25 * S
    return (float(qx), float(qy), float(qz), float(qw))


def f_translation(pos):
    out = np.eye(4)
    out[:3, 3] = pos
    return out

def f_rot_trans(rot, pos):
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3,  3] = pos
    return out

def draw_gmm(vis : ROSVisualizer, namespace, gmm, dimensions=None):
    dimensions = list(gmm.semantic_dims().keys())

    for d in dimensions:
        if '|' in d:
            dims_in, dims_out = d.split('|')
            dims_in  = gmm.semantic_dims()[dims_in]
            dims_out = gmm.semantic_dims()[dims_out]

            for k, (mu_k, sigma_k) in enumerate(zip(gmm.mu(dims_in), gmm.sigma(dims_in, dims_out))):
                print(sigma_k)
                w, v = np.linalg.eig(sigma_k)
                print(f'{k}:\n{w}\n{v}\n{np.sqrt((v**2).sum(axis=0))}\nQ: {real_quat_from_matrix(v)} \n -------')
                
                vis.draw_ellipsoid(namespace, f_rot_trans(v, mu_k), w)
        else:
            dims = gmm.semantic_dims()[d]

            for k, (mu_k, sigma_k) in enumerate(zip(gmm.mu(dims), gmm.sigma(dims, dims))):
                print(sigma_k)
                w, v = np.linalg.eig(sigma_k)
                print(f'{k}:\n{w}\n{v}\n{np.sqrt((v**2).sum(axis=0))}\nQ: {real_quat_from_matrix(v)} \n -------')
                
                vis.draw_ellipsoid(namespace, f_rot_trans(v, mu_k), w)


if __name__ == '__main__':
    parser = ArgumentParser(description='RVIZ visualization of GMMs')
    parser.add_argument('gmm', help='GMM to visualize')
    # parser.add_argument('trajectories', nargs='+', help='Trajectories to load as starting points and references.')
    # parser.add_argument('--show-all', action='store_true', help='Show all rollouts at once or one-by-one.')
    # parser.add_argument('--steps', default=300, type=int, help='Number of steps to rollout.')
    # parser.add_argument('--action-freq', default=30, type=int, help='Prediction interval in Hz.')
    parser.add_argument('--dims', default=None, nargs='*', help='Variances to draw')
    parser.add_argument('--print', action='store_true', help='Print GMM information')
    args = parser.parse_args()

    # trajs = unpack_trajectories(args.trajectories, 
    #                             [np.load(t, allow_pickle=True) for t in args.trajectories])
    gmm = GMM.load_model(args.gmm)

    if args.print:
        print(f'GMM ({gmm.n_priors}, {gmm.n_dims})\n  Prior: {gmm.pi()}\n  Means: \n {gmm.mu()}\n  Cvar: \n {gmm.sigma()}')


    rospy.init_node('bopt_gmm_visualizer')

    vis = ROSVisualizer('vis')

    vis.begin_draw_cycle('gmms')
    draw_gmm(vis, 'gmms', gmm, ['position'])
    vis.render('gmms')

    # for _, _, _, t in trajs:
    #     gmm_traj = rollout(gmm, t[0, :gmm.n_dims // 2], args.steps, 1 / args.action_freq)
    #     vis.draw_strip('gmm_path', np.eye(4), 0.005, gmm_traj[:, :3], r=0.8, g=0.5)
    #     vis.draw_strip('demo', np.eye(4), 0.005, t[:, :3], r=0, b=0.9)
    #     vis.render()

    #     if not args.show_all:
    #         bla = input('Press enter to see next GMM')
    #         vis.begin_draw_cycle('gmm_path', 'demo')
    
    rospy.sleep(0.1) # For topics are stupid
