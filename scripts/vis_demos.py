import numpy as np
import matplotlib.pyplot as plt
import operator

from argparse  import ArgumentParser
from functools import reduce

from bopt_gmm.utils   import unpack_trajectories, \
                             gauss_smoothing


def plot_demonstrations(axes, dim_names, groups, data, plot_name='Trajectory', alignment='start'):
    indices = list(range(data.shape[0])) if alignment == 'start' else list(range(-data.shape[0], 0))
    
    for idx, (name, d_data, ax) in enumerate(zip(dim_names, data.T, axes)):
        if idx == 0:
            ax.set_title(plot_name)
        
        if idx == len(axes) - 1:
            ax.set_xlabel('Step')
        else:
            ax.set_xticklabels([])


        ax.plot(indices, d_data)
        ax.set_ylabel(name)
        ax.grid(True)
    
    # for g in groups:
    #     g_data = data[:,g] #  np.take(data, g, axis=0)
    #     g_lim  = np.asarray((g_data.min(), g_data.max()))
    #     g_lim  = np.asarray((-0.55, 0.55)) * (g_lim[1] - g_lim[0]) + g_lim.mean() 
    #     for a in np.take(axes, g):
    #         a.set_ylim(g_lim)


def plot_fig(trajectories, axes_dim_in=(6, 1.5), plot_path=None, overlay=False, alignment='start'):
    rows = max(len(dim_names) for _, dim_names, _, _ in trajectories)
    cols = len(trajectories) if not overlay else 1

    fig  = plt.figure(figsize=(axes_dim_in[0] * cols, axes_dim_in[1] * rows))
    gs   = fig.add_gridspec(rows, cols)
    
    if overlay:
        axes = [fig.add_subplot(gs[r:r+1, 0]) for r in range(rows)]

    for c, (fp, dim_names, groups, data) in enumerate(trajectories):
        if not overlay:
            axes = [fig.add_subplot(gs[r:r+1, c:c+1]) for r in range(len(dim_names))]
        plot_demonstrations(axes, dim_names, groups, data, fp, alignment=alignment)

    fig.tight_layout()

    if plot_path is not None:
        plot_path = f'{plot_path}.png' if plot_path[-4:].lower() not in {'.png', '.jpg', '.jpeg'} else plot_path
        fig.savefig(plot_path)
    else:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    parser = ArgumentParser(description='SEDS test')
    parser.add_argument('trajectories', nargs='+', help='List of trajectories to fit to')
    parser.add_argument('--dim-names', default=None, nargs='*', help='Override of dim names for unstructured trajectories')
    parser.add_argument('--draw-all', default=False, action='store_true', help='Draw all into one plot.')
    parser.add_argument('--overlay', default=False, action='store_true', help='Draw all paths over one another.')
    parser.add_argument('--out', default=None, help='Path to save plot to.')
    parser.add_argument('--alignment', default='start', choices=['start', 'end'], help='Align trajectories at the beginning or the end.')
    parser.add_argument('--smoothing', type=int, default=0, help='Apply smoothing to the data by blending n consecutive steps.')
    args = parser.parse_args()

    tfs = [np.load(t, allow_pickle=True) for t in args.trajectories]

    trajs = unpack_trajectories(args.trajectories, tfs)

    if args.smoothing > 0:
        trajs = [(fp, dim_names, groups, gauss_smoothing(data, args.smoothing)) for fp, dim_names, groups, data in trajs]

    if not args.draw_all and not args.overlay:
        for t in trajs:
            plot_fig([t], plot_path=args.out, overlay=args.overlay, alignment=args.alignment)
    else:
        plot_fig(trajs, plot_path=args.out, overlay=args.overlay, alignment=args.alignment)
