import numpy as np

from argparse import ArgumentParser

from drlfads.lpvDS.seds import SEDS_MATLAB, \
                               gen_seds_data_from_trajectories, \
                               print_seds_data_analysis,  \
                               seds_data_prep_last_step

from drlfads.utils.bopt_utils import unpack_trajectories

if __name__ == '__main__':
    parser = ArgumentParser(description='SEDS test')
    parser.add_argument('seds_path',                                help='Root of SEDS MATLAB implementation')
    parser.add_argument('trajectories',  nargs='+',                 help='List of trajectories to fit to')
    parser.add_argument('--max-iter',    default=500,   type=int,   help='Maximum number of iterations for SEDS')
    parser.add_argument('--tol-cutting', default=0.1,   type=float, help='Velocity cut-off')
    parser.add_argument('--objective',   default='likelihood', type=str,   help='Velocity cut-off', choices=['mse', 'likelihood', 'direction'])
    parser.add_argument('--n-priors',    default=3,     type=int,   help='Number of GMM priors')
    parser.add_argument('--out',         default='seds_model',      help='File name of the new model.')
    parser.add_argument('--action-freq', default=30,    type=float, help='Action frequency.')
    parser.add_argument('--use-force',   action='store_true',       help='Generate GMM using forces')
    args = parser.parse_args()

    if args.out.lower()[-4:] != '.npy':
        args.out = f'{args.out}.npy'

    trajs = unpack_trajectories(args.trajectories, [np.load(t, allow_pickle=True) for t in args.trajectories], {'position', 'force'})
    trajs = [data for _, _, _, data in trajs]

    if not args.use_force:
        trajs = [d[:,:3] for d in trajs]

    dt = 1 / args.action_freq

    x0, xT, data, o_idx = gen_seds_data_from_trajectories(trajs, dt)

    print_seds_data_analysis(x0, xT, data, o_idx)

    x0, xT, data, o_idx = seds_data_prep_last_step(x0, xT, data, o_idx)

    # trajs = np.load(args.trajectories, allow_pickle=True)
    # print(len(trajs))
    # # Remove force again
    # x0, xT, data, o_idx = gen_trajectory_from_transitions(trajs, 1 / 100, args.use_force) # Force)

    # # exit()
    seds = SEDS_MATLAB(args.seds_path)

    gmm = seds.fit_model(x0, xT, data, o_idx, args.n_priors, args.objective, dt, args.tol_cutting, args.max_iter)
    gmm.save_model(args.out)
