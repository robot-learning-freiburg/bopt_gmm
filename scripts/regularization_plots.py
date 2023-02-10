import matplotlib.pyplot as plt
import math
import pandas as pd

from argparse import ArgumentParser
from pathlib  import Path

if __name__ == '__main__':
    parser = ArgumentParser(description='Generates scatter plots for a given summary file.')
    parser.add_argument('summary', help='Summary file')
    parser.add_argument('--out', default=None, help='Name of output image.')
    parser.add_argument('--pw', default=3, type=float, help='Plot width in inches')
    parser.add_argument('--ph', default=1.5, type=float, help='Plot height in inches')
    args = parser.parse_args()

    if args.out is None:
        p = Path(args.summary)
        args.out = f'{p.parent}/{p.name[:-4]}.png'
    elif args.out.lower()[-4:] != '.png':
        args.out = f'{args.out}.png'

    df = pd.read_csv(args.summary)

    try:
        metric_index = list(df.columns).index('demo_base') + 1
    except ValueError:
        metric_index = list(df.columns).index('demo_base') + 1
    
    n_plots = len(df.columns) - metric_index

    cols = int(math.sqrt(n_plots))
    rows = int(math.ceil(n_plots / cols))

    fig  = plt.figure(figsize=(args.pw * cols, args.ph * rows))
    gs   = fig.add_gridspec(rows, cols)

    for i, metric in enumerate(df.columns[metric_index:]):
        data = df[[metric, 'accuracy']].to_numpy().T

        iy = i // cols
        ix = i % cols
        ax = fig.add_subplot(gs[iy:iy+1, ix:ix+1])

        ax.set_title(metric)
        ax.grid(True)
        
        if ix == 0:
            ax.set_ylabel('Accuracy')
        
        ax.scatter(data[0], data[1], marker='.')
        ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(args.out)
