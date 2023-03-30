import numpy as np
import torch

from argparse import ArgumentParser
from tqdm     import tqdm

from bopt_gmm.baselines import LSTMPolicy, \
                               LSTMPolicyConfig, \
                               MLP
from bopt_gmm.utils     import unpack_trajectories, \
                               calculate_trajectory_velocities



if __name__ == '__main__':
    parser = ArgumentParser(description='Generate GMMs from trajectory data')
    parser.add_argument('trajectories',  nargs='+',                  help='List of trajectories to fit to.')
    parser.add_argument('--policy',      default='lstm', type=str,   help='Choice of policy model.', choices=['lstm'])
    parser.add_argument('--max-iter',    default=500,    type=int,   help='Maximum number of training episodes.')
    parser.add_argument('--lr',          default=1e-3,   type=float, help='Learning rate.')
    parser.add_argument('--embedding',   default=6,      type=int,   help='Embeding size.')
    parser.add_argument('--out',         default='lstm_model',       help='File name of the new model.')
    parser.add_argument('--action-freq', default=30,     type=float, help='Action frequency.')
    parser.add_argument('--modalities',  default=['position'], nargs='+', help='Modalities on which to train GMM')
    parser.add_argument('--mlp-arch',    default=[256, 256],   nargs='+', help='List of MLP layer sizes.')
    parser.add_argument('--action-dim',  default=3,      type=int,   help='Action dimension.')
    parser.add_argument('--split',       default=0.8,    type=float, help='Training eval split.')
    args = parser.parse_args()

    if args.out.lower()[-4:] != '.npz':
        args.out = f'{args.out}.npz'


    device = torch.device('cuda:0') if torch.has_cuda else torch.cuda.device('cpu')

    trajs = unpack_trajectories(args.trajectories, 
                                [np.load(t, allow_pickle=True) for t in args.trajectories], 
                                args.modalities)

    trajs = [data for _, _, _, data in trajs]

    dt = 1 / args.action_freq

    data = [torch.tensor(calculate_trajectory_velocities(t, dt), dtype=torch.float32).unsqueeze(0).to(device) for t in trajs]

    lstm_model = LSTMPolicy(MLP(data[0].shape[2] // 2, 
                                args.embedding,
                                hidden_dim=args.mlp_arch),
                            args.action_dim,
                            LSTMPolicyConfig(args.lr, 3e-6),
                            device=device)

    t_split = int(len(data) * args.split)
    t_data  = data[:t_split]
    e_data  = data[t_split:]

    for ep in tqdm(range(args.max_iter), desc='Training policy'):
        for t in t_data:
            obs    = t[:,:,:t.shape[2] // 2]
            action = t[:,:,t.shape[2] // 2:t.shape[2] // 2 + args.action_dim]
        
        losses = []
        for t in e_data:
            obs    = t[:,:,:t.shape[2] // 2]
            action = t[:,:,t.shape[2] // 2:t.shape[2] // 2 + args.action_dim]

            losses.append(lstm_model.update_params(obs, action)['loss'].detach().cpu().numpy())
            
            print(f'Policy loss episode {ep}: {np.mean(losses)}')

    lstm_model.save_model(args.out)
