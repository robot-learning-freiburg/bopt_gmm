import glob
import numpy  as np
import pandas as pd
import regex

from argparse import ArgumentParser
from pathlib  import Path
from tqdm     import tqdm

import bopt_gmm.bopt.regularization as reg

from bopt_gmm.utils import unpack_trajectories
from bopt_gmm.gmm   import GMM, GMMCart3DForce, rollout, GMMCart3D


if __name__ == '__main__':
    parser = ArgumentParser(description='Calculates the regularization values for a models from a results file compared against their base models.')
    parser.add_argument('results', help='Results file to process')
    parser.add_argument('trajectories', nargs='+', help='Trajectories used as sample for regularization. Trajectory paths are parsed for components and noise.')
    parser.add_argument('--out', default='reg_summary.csv', help='File to write the results to.')
    args = parser.parse_args()

    df = pd.read_csv(args.results)

    if 'model' not in df:
        print(f'Dataframe is missing a "model" column')
        exit(-1)
    elif 'base' not in df:
        print(f'Dataframe is missing a "base" column')
        exit(-1)
    
    bases = {} 
    for b in set(df.base):
        try:
            bases[b] = GMM.load_model(b)
        except Exception:
            pass

    demo_bases = None
    if 'demo_base' in df:
        demo_bases = {b: GMM.load_model(b) for b in set(df.demo_base)}

    trajectories = {}
    for p_traj in args.trajectories:
        p_traj = Path(p_traj)
        noise  = 0.0  # int(regex.findall(r"_n\d\d_", str(p_traj.name))[0][2:-1]) * 0.01
        components = 3 # int(regex.findall(r"_\d+p_", str(p_traj.name))[0][1:-2])

        if components not in trajectories:
            trajectories[components] = {}
        
        trajectories[components][noise] = np.vstack([pos for _, _, _, pos in unpack_trajectories([p_traj], 
                                                                                                 [np.load(p_traj, allow_pickle=True)], 
                                                                                                 ['position'])])
        
        # Hacking the range of noises
        for n in range(0, 6):
            trajectories[components][n * 0.01] = trajectories[components][0.0]
    
    # Hacking the range of components
    trajectories[5] = trajectories[components]
    trajectories[7] = trajectories[components]


    f_regs = [reg.f_joint_prob, reg.f_mean_prob, reg.f_kl, reg.f_jsd, reg.f_dot]

    new_cols = 'j_prob mean_prob kl jsd dot'.split(' ')
    if demo_bases is not None:
        new_cols += [f'db_{c}' for c in new_cols]


    sub_fields  = ['components', 'noise', 'model', 'base']
    if demo_bases is not None:
        sub_fields += ['demo_base']

    data = df[sub_fields].to_numpy()
    if demo_bases is None:
        data = np.hstack((data, np.zeros((data.shape[0], 1))))

    # print(data)

    values = []
    for c, n, m, b, db in tqdm(data, desc='Calculating regularizations...'):
        gmm = GMM.load_model(m)
        
        if b not in bases:
            continue

        base_gmm = bases[b]
        positions = trajectories[c][n]

        out = [f(gmm, base_gmm, positions) for f in f_regs]
    
        if demo_bases is not None:
            demo_gmm = demo_bases[db]
            out += [f(gmm, demo_gmm, positions) for f in f_regs]

        values.append(out)

    df_v = pd.DataFrame(values, columns=new_cols)

    df.join(df_v).to_csv(args.out, index=False)

    

    