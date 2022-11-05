import numpy as np
import os

from argparse import ArgumentParser


from bopt_gmm.gmm.generation import SEDS_MATLAB, \
                                    gen_seds_data_from_trajectories, \
                                    seds_data_prep_last_step, \
                                    print_seds_data_analysis

from bopt_gmm.utils import unpack_trajectories

if __name__ == '__main__':
    if 'SEDS_PATH' not in os.environ:
        print('Need environment variable SEDS_PATH pointing to the root of the SEDS repo')
        exit()

    parser = ArgumentParser(description='SEDS test')
    parser.add_argument('trajectories',  nargs='+',                 help='List of trajectories to fit to')
    parser.add_argument('--max-iter',    default=500,   type=int,   help='Maximum number of iterations for SEDS')
    parser.add_argument('--tol-cutting', default=0.1,   type=float, help='Velocity cut-off')
    parser.add_argument('--objective',   default='likelihood', type=str,   help='Velocity cut-off', choices=['mse', 'likelihood', 'direction'])
    parser.add_argument('--n-priors',    default=3,     type=int,   help='Number of GMM priors')
    parser.add_argument('--out',         default='seds_model',      help='File name of the new model.')
    parser.add_argument('--action-freq', default=30,    type=float, help='Action frequency.')
    parser.add_argument('--modalities',  default=['position'], nargs='+', help='Modalities on which to train GMM')
    args = parser.parse_args()

    if args.out.lower()[-4:] != '.npy':
        args.out = f'{args.out}.npy'

    trajs = unpack_trajectories(args.trajectories, 
                                [np.load(t, allow_pickle=True) for t in args.trajectories], 
                                args.modalities)
    trajs = [data for _, _, _, data in trajs]

    dt = 1 / args.action_freq

    x0, xT, data, o_idx = gen_seds_data_from_trajectories(trajs, dt)

    print_seds_data_analysis(x0, xT, data, o_idx)

    x0, xT, data, o_idx = seds_data_prep_last_step(x0, xT, data, o_idx)

    # trajs = np.load(args.trajectories, allow_pickle=True)
    # print(len(trajs))
    # # Remove force again
    # x0, xT, data, o_idx = gen_trajectory_from_transitions(trajs, 1 / 100, args.use_force) # Force)

    # # exit()
    seds = SEDS_MATLAB(os.environ['SEDS_PATH'])

    gmm = seds.fit_model(x0, xT, data, o_idx, args.n_priors, args.objective, dt, args.tol_cutting, args.max_iter)
    gmm.save_model(args.out)
