import matplotlib.pyplot   as plt
import math
import matplotlib.gridspec as gridspec
import numpy  as np
import pandas as pd

from argparse  import ArgumentParser
from itertools import product
from pathlib   import Path



COLORS = [['#12DB00', '#70F20C', '#B8E80C', '#0CF23E', '#0CE879'],
          ['#DB6300', '#F2520C', '#E82F0C', '#F2940C', '#E8A40C'],
          ['#42CEF5', '#31DED5', '#37FAB9', '#318BDE', '#3769FA'],
          ['#F4D221', '#DEA814', '#FAA717', '#DED114', '#DFFA17'],
          ['#F51D4F', '#DE10A7', '#DD12FA', '#DE2010', '#FA4612'],
          ['#7C15F5', '#3209DE', '#0A1CFA', '#A609DE', '#FA0AF5'],
          ['#1B3EF5', '#0D67DE', '#0FB2FA', '#220DDE', '#670FFA']]


def draw_var_plot(ax, df, color, prefix=''):
    d     = np.vstack([v for v in [df.accuracy[df.base == b].to_numpy() for b in df.base.unique()] if v.shape[0] == df.tsteps.max()]).T
    means_ps = np.mean(d, axis=1)
    stds_ps  = np.std(d, axis=1)
    x_coords = list(range(df.tsteps.max()))
    ax.plot(x_coords, means_ps, label=f'{prefix} Steps: {df.tsteps.max()}', c=color)
    ax.fill_between(x_coords, means_ps-stds_ps, means_ps+stds_ps, alpha=0.3, facecolor=color)


def draw_metrics_subplot(fig, gs, df,  w_inner, h_inner, prefix=''):
    s_gs   = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs, wspace=w_inner, hspace=h_inner)
    coords = product(range(2), range(2))

    criteria = [None, 'weights', 'means', 'sigma']

    for cr, (x, y) in zip(criteria, coords):
        ax = fig.add_subplot(s_gs[y, x])

        ax.grid(True)

        if x == 0 and y == 0:
            ax.set_title(prefix)

        if x == 0:
            ax.set_ylabel('Accuracy')
        else:
            ax.set(yticklabels=[])

        if y == 1:
            ax.set_xlabel('Optimizer Steps')
        else:
            ax.set(xticklabels=[])

        ax.set_ylim(-0.05, 1.05)

        if cr is not None:
            for cs, v in zip(COLORS[1:], df[cr].unique()):
                df_v = df[df[cr] == v]
                for c, n_eval in zip(cs, df_v.tsteps.unique()):
                    draw_var_plot(ax, df_v[df_v.tsteps == n_eval], c, prefix=f'{cr}: {v}')
        else:
            for c, n_eval in zip(COLORS[0], df.tsteps.unique()):
                draw_var_plot(ax, df[df.tsteps == n_eval], c)

        ax.legend()


if __name__ == '__main__':
    parser = ArgumentParser(description='Generates scatter plots for a given summary file.')
    parser.add_argument('summary', help='Summary file')
    parser.add_argument('--out', default=None,          help='Name of output image.')
    parser.add_argument('--pw',  default=6, type=float, help='Plot width in inches')
    parser.add_argument('--ph',  default=3, type=float, help='Plot height in inches')
    parser.add_argument('--ac-filter',  default=0.0, type=float, help='Accuracy filter')
    args = parser.parse_args()

    if args.out is None:
        p = Path(args.summary)
        args.out = f'{p.parent}/{p.name[:-4]}.png'
    elif args.out.lower()[-4:] != '.png':
        args.out = f'{args.out}.png'

    df = pd.read_csv(args.summary)

    base_filters = df[df.accuracy >= args.ac_filter].base.unique()

    df = df[df.base.isin(base_filters)]

    optimizers   = df.optimizer.unique()

    opt_dfs = {o: df[df.optimizer == o] for o in optimizers}

    cols = len(opt_dfs)
    rows = len(df.noise.unique())

    fig  = plt.figure(figsize=(args.pw * cols * 2, args.ph * rows * 2))
    
    master_gs = gridspec.GridSpec(len(df.noise.unique()), len(opt_dfs), figure=fig)

    for m_x, (o, df) in enumerate(sorted(opt_dfs.items())):
        for m_y, n in enumerate(sorted(df.noise.unique())):
            
            df_n = df[df.noise == n]

            gs = master_gs[m_y, m_x]

            draw_metrics_subplot(fig, gs, df_n, 0.05, 0.05, f'Opt: {o} Noise: {n}')

    fig.tight_layout()
    fig.savefig(args.out)
