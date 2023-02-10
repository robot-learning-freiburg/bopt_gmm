import matplotlib.pyplot as plt
import math
import numpy  as np
import pandas as pd
import regex

from argparse         import ArgumentParser
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pandas.api.types import is_numeric_dtype
from pathlib          import Path

from bopt_gmm.logging import MP4VideoLogger

def __groupby_slice( _grp, start=0, stop=None, step=1):
    '''
    Applies a slice to a GroupBy object
    '''
    print(f'{start}, {stop}')
    return _grp.apply( lambda _df : _df.iloc[start:stop:step]).reset_index(drop=True)

pd.core.groupby.GroupBy.slice = __groupby_slice


numerical_types = {float, int, 
                   np.float16, np.float32, np.float64, np.float128,
                   np.int8, np.int16, np.int32, np.int64,
                   }

if __name__ == '__main__':
    parser = ArgumentParser(description='Generates scatter plots for a given summary file.')
    parser.add_argument('summary', help='Summary file')
    parser.add_argument('--out', default=None, help='Name of output image.')
    parser.add_argument('--pw', default=3, type=float, help='Plot width in inches')
    parser.add_argument('--ph', default=1.5, type=float, help='Plot height in inches')
    parser.add_argument('--sl', default=0, type=int, help='Lower bound of evals to slice from.')
    parser.add_argument('--su', default=1000000, type=int, help='Upper bound of evals to slice from.')
    parser.add_argument('--evolution', action='store_true', help='Create an animation instead of a single image of the evolution of the plot.')
    parser.add_argument('--ws', default=1000000, type=int, help='Window size of the evolving animation.')
    args = parser.parse_args()

    if args.evolution:
        if args.out is None:
            p = Path(args.summary)
            args.out = f'{p.parent}/{p.name[:-4]}.mp4'
        elif args.out.lower()[-4:] != '.mp4':
            args.out = f'{args.out}.mp4'
    else:
        if args.out is None:
            p = Path(args.summary)
            args.out = f'{p.parent}/{p.name[:-4]}.png'
        elif args.out.lower()[-4:] != '.png':
            args.out = f'{args.out}.png'

    df = pd.read_csv(args.summary)

    pattern = r"_(ei|pi|gph)_(\d+)_(\d+)_(lhs|halton|random|sobol)_(auto|sampling|lbfgs)"

    try:
        metric_index = list(df.columns).index('demo_base') + 1
    except ValueError:
        metric_index = list(df.columns).index('demo_base') + 1
    

    hp_data   = []
    hp_header = 'acq_func acq_opt init_gen p_range mu_range'.split(' ')
    
    for p in df.path:
        func, pr, mr, sampling, opt = regex.findall(pattern, p)[0]
        pr = int(pr) * 0.01
        mr = int(mr) * 0.01
        hp_data.append([func, opt, sampling, pr, mr])

    def custom_series_mean(series):
        if is_numeric_dtype(series):
            return series.mean()
        return series.iloc[0]

    df_full = df.join(pd.DataFrame(hp_data, columns=hp_header))

    args.su = min(args.su, len(df_full))

    if args.evolution:
        ranges = [(max(args.sl, x+1 - args.ws), x+1) for x in range(args.sl, args.su)]
    else:
        ranges = [(args.sl, args.su)]

    n_plots = len(hp_header)

    cols = int(math.sqrt(n_plots))
    rows = int(math.ceil(n_plots / cols))

    if args.evolution:
        img_shape = (int(args.ph * rows * 100), int(args.pw * cols * 100), 3)
        writer = MP4VideoLogger(Path(args.out).parent, 
                                Path(args.out).name,
                                (img_shape[1], img_shape[0]), 10)
        print(img_shape)

    for rl, ru in ranges:
        fig  = plt.figure(figsize=(args.pw * cols, args.ph * rows))
        if args.evolution:
            canvas = FigureCanvas(fig)

        gs   = fig.add_gridspec(rows, cols)

        df_sorted = df_full.assign(group=df_full[hp_header].apply(frozenset, axis=1)).sort_values('eval', ascending=True).groupby('group').slice(rl, ru).groupby('group').aggregate(custom_series_mean)

        for i, metric in enumerate(hp_header):
            unique_values = sorted(df_sorted[[metric, 'accuracy']][metric].unique())
            data = [df_sorted['accuracy'][df_sorted[metric] == uv].to_numpy() for uv in unique_values]

            iy = i // cols
            ix = i % cols
            ax = fig.add_subplot(gs[iy:iy+1, ix:ix+1])

            ax.set_title(metric)
            ax.grid(True)
            
            if ix == 0:
                ax.set_ylabel('Accuracy')
            
            positions = list(range(len(unique_values)))
            ax.violinplot(data, positions=positions, vert=True, points=20, showmeans=True)
            
            ax.set_xticks(positions)
            if type(unique_values[0]) != str:
                unique_values = [f'{uv:.2f}' for uv in unique_values]

            ax.set_xticklabels(unique_values)
            # ax.scatter(data[0], data[1], marker='.')
            ax.set_ylim(-0.05, 1.05)

        fig.tight_layout()
        if args.evolution:
            canvas.draw()       # draw the canvas, cache the renderer
            image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(img_shape)[:,:,::-1]
            writer.write_image(image)
        else:
            fig.savefig(args.out)

        plt.close(fig)

    if args.evolution:
        del writer
