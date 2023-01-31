import glob
import numpy  as np
import pandas as pd

from argparse import ArgumentParser
from pathlib  import Path

import bopt_gmm.bopt.regularization as reg

from bopt_gmm.utils import unpack_trajectories
from bopt_gmm.gmm   import GMMCart3DForce, rollout, GMMCart3D

if __name__ == '__main__':
    parser = ArgumentParser(description='Calculates the regularization values for a models and trajectories in a folder.')
    parser.add_argument('path', help='Directory to process.')
    parser.add_argument('trajectories', nargs='+', help='Trajectories used as sample for regularization.')
    args = parser.parse_args()

    pattern = f'{args.path}/gmm_*.npy'

    positions = [pos for _, _, _, pos in unpack_trajectories(args.trajectories, 
                                                             [np.load(t, allow_pickle=True) for t in args.trajectories], 
                                                             ['position', 'force'])]
    # positions = np.vstack(positions)

    gmms = []

    if Path(args.path).is_dir():
        for p in glob.glob(pattern):
            p = Path(p)
            
            if p.name == 'gmm_base.npy':
                base_gmm = GMMCart3D.load_model(p)
            else:
                gmms.append((p, GMMCart3D.load_model(p)))
    else:
        gmms = [GMMCart3DForce.load_model(p)]
        base_gmm = gmms[0]

    f_regs = [reg.f_joint_prob, reg.f_mean_prob, reg.f_kl, reg.f_jsd, reg.f_dot]

    values = []

    gmms.insert(0, (Path('gmm_base.npy'), base_gmm))

    positions = np.vstack([rollout(gmms[0][1], p[0][:len(gmms[0][1].state_dim)], True) for p in positions])

    for gmm_path, gmm in gmms:
        values.append([f(gmm, base_gmm, positions) for f in f_regs])
    
    print(''.join([' ' * 15] + [f'{f.__name__:>15}' for f in f_regs]))
    for (gp, _), values in zip(gmms, values):
        print(''.join([f'{gp.name:>14} '] + [f' {v:>14.5f}' for v in values]))

    