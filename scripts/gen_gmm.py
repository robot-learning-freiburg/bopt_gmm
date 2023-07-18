import numpy as np
import os
import bopt_gmm.gmm as libgmm


from argparse import ArgumentParser


from bopt_gmm.gmm.generation import SEDS_MATLAB, \
                                    gmm_fit_em, \
                                    gen_seds_data_from_trajectories, \
                                    seds_data_prep_last_step, \
                                    print_seds_data_analysis

from bopt_gmm.utils import unpack_trajectories, \
                           calculate_trajectory_velocities, \
                           normalize_trajectories

MODEL_MAP = {'position' : libgmm.GMMCart3D,
             '_'.join(sorted(['position', 'force']))  : libgmm.GMMCart3DForce,
             '_'.join(sorted(['position', 'torque'])) : libgmm.GMMCart3DTorque}

if __name__ == '__main__':
    parser = ArgumentParser(description='Generate GMMs from trajectory data')
    parser.add_argument('trajectories',  nargs='+',                  help='List of trajectories to fit to')
    parser.add_argument('--generator',   default='seds', type=str,   help='Choice of generation technique', choices=['seds', 'em'])
    parser.add_argument('--max-iter',    default=500,    type=int,   help='Maximum number of iterations for SEDS')
    parser.add_argument('--tol-cutting', default=0.1,    type=float, help='Velocity cut-off')
    parser.add_argument('--objective',   default='likelihood', type=str,   help='Velocity cut-off', choices=['mse', 'likelihood', 'direction'])
    parser.add_argument('--n-priors',    default=3,      type=int,   help='Number of GMM priors')
    parser.add_argument('--n-init',      default=1,      type=int,   help='Number of re-initializations for EM generator.')
    parser.add_argument('--out',         default='seds_model',       help='File name of the new model.')
    parser.add_argument('--action-freq', default=30,     type=float, help='Action frequency.')
    parser.add_argument('--normalize',   action='store_true',        help='Normalize additional modalities to the space of positions.')
    parser.add_argument('--modalities',  default=['position'], nargs='+', help='Modalities on which to train GMM')
    parser.add_argument('--prior',       default=None, help='Prior GMM to use for initializing optimizer.')
    parser.add_argument('--action-dim',  default=None, help='Override the velocity dimension with explicit action dimension.')
    args = parser.parse_args()

    if args.generator == 'seds':
        if 'SEDS_PATH' not in os.environ:
            print('Need environment variable SEDS_PATH pointing to the root of the SEDS repo')
            exit()

    if args.out.lower()[-4:] != '.npy':
        args.out = f'{args.out}.npy'

    trajs = unpack_trajectories(args.trajectories, 
                                [np.load(t, allow_pickle=True) for t in args.trajectories], 
                                args.modalities)
    
    if args.action_dim is not None:
        actions = unpack_trajectories(args.trajectories, 
                                    [np.load(t, allow_pickle=True) for t in args.trajectories], 
                                    [args.action_dim])
        actions = np.vstack([data[1:] for _, _, _, data in trajs])
    else:
        actions = None

    if args.normalize and 'position' in args.modalities and len(args.modalities) > 0:
        trajs, group_norms = normalize_trajectories(trajs, 'position')
    else:
        group_norms = None

    trajs = [data for _, _, _, data in trajs]

    dt = 1 / args.action_freq

    model_kwargs = {}
    try:
        gmm_type = MODEL_MAP['_'.join(sorted(args.modalities))]
        model_kwargs = gmm_type.model_kwargs_from_groups(group_norms, args.modalities)
    except KeyError:
        libgmm.GMMCart3DJS.model_kwargs_from_groups(group_norms, args.modalites)


    if args.generator == 'seds':
        trajs = [t * 1000.0 for t in trajs]
        x0, xT, data, o_idx = gen_seds_data_from_trajectories(trajs, dt)

        print_seds_data_analysis(x0, xT, data, o_idx)

        x0, xT, data, o_idx = seds_data_prep_last_step(x0, xT, data, o_idx)

        # trajs = np.load(args.trajectories, allow_pickle=True)
        # print(len(trajs))
        # # Remove force again
        # x0, xT, data, o_idx = gen_trajectory_from_transitions(trajs, 1 / 100, args.use_force) # Force)

        # # exit()
        seds = SEDS_MATLAB(os.environ['SEDS_PATH'])

        model_kwargs['general_scale'] = 1000.0

        gmm = seds.fit_model(x0, xT, data, o_idx, args.n_priors, args.objective, dt, 
                             args.tol_cutting, args.max_iter, 
                             gmm_type=gmm_type, model_kwargs=model_kwargs)
    elif args.generator == 'em':
        if args.prior is not None:
            prior_gmm = libgmm.gen_prior_gmm(gmm_type, libgmm.GMM.load_model(args.prior), model_kwargs=model_kwargs)
        else:
            prior_gmm = None

        traj_width = trajs[0].shape[1]
        data       = np.vstack([calculate_trajectory_velocities(t, dt) for t in trajs])
        if actions is not None:
            data[:,traj_width:traj_width + actions.shape[1]] = actions


        gmm  = gmm_fit_em(args.n_priors, data, 
                          max_iter=args.max_iter, 
                          tol=args.tol_cutting, 
                          n_init=args.n_init,
                          gmm_type=gmm_type,
                          prior_gmm=prior_gmm,
                          model_kwargs=model_kwargs)

    gmm.save_model(args.out)

    if group_norms is not None:
        print('\n-----\nGroup norms:\n  {}'.format('\n  '.join([f'{k}: {v}' for k, v in group_norms.items()])))