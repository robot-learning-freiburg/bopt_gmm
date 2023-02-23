import glob
import pandas as pd 
import numpy  as np
import regex
import yaml


from argparse import ArgumentParser
from pathlib  import Path
from tqdm     import tqdm

from bopt_gmm.gmm import GMM


if __name__ == '__main__':
    parser = ArgumentParser(description='Summarize eval stats from a collection of folders.')
    parser.add_argument('dirs', nargs='+', help='Directory to process.')
    parser.add_argument('--out', default='summary.csv', help='Name of file containing the summary.')
    parser.add_argument('--base-models', default=None, help='Path to a yaml file that collects the geometric base models for each number of components.')
    args = parser.parse_args()


    if args.base_models is not None:
        with open(args.base_models, 'r') as f:
            base_models = yaml.load(f)
        
        p = Path(args.base_models).absolute().parent
        base_models = {n: f'{p}/{m}' for n, m in base_models.items()}


    header  = 'path components noise force tsteps opt weights means sigma eval accuracy model base'.split(' ')
    if args.base_models is not None:
        header += ['demo_base']
    
    summary = []

    r_pattern = r'(_(\d+))?(_f|)_(\d)_(([a-z]+_)+|)n(\d\d)_([a-z]+)_(\d+)_(\d+)_([a-z]+)_([a-z]+)'

    for d in tqdm(args.dirs, desc='Processing directories'):
        pattern = f'{d}/eval_*_ic.csv'

        for p in glob.glob(pattern):
            p = Path(p)
            
            try:
                eval_num = int(p.name[5:-7])
            except ValueError as e:
                print(f'Error raised while processing file {p}:\n{e}')
                exit(0)

            try:
                df = pd.read_csv(p)
            except pd.errors.EmptyDataError:
                print(f'Results file {p} is empty. Skipping...')
                continue

            try:
                _, train_steps, force, components, optim_groups, _, noise, _, prior, m, _, func = regex.findall(r_pattern, str(Path(p).parent.name))[0]
            except (ValueError, IndexError) as e:
                print(f'{p}\n{str(Path(p).parent.name)}\n{regex.findall(pattern, str(Path(p).parent.name))}:\n{e}')
                exit(0)

            train_steps  = int(train_steps) if train_steps != '' else ''
            force        = False if force == '' else True
            components   = int(components)
            optim_groups = [x for x in optim_groups.split('_') if x != '_']
            noise        = int(noise) * 0.01
            prior        = int(prior) * 0.01
            m            = int(m) * 0.01
            s            = 0.0

            # noise_vectors = np.vstack((df.position_noise_x.array, 
            #                            df.position_noise_y.array,
            #                            df.position_noise_z.array)).T
            # noise = np.sqrt((noise_vectors ** 2).sum(axis=1)).mean()

            model = f'{p.parent}/models/gmm_{eval_num}.npy'
            base  = f'{p.parent}/models/gmm_base.npy'

            if args.base_models is None:
                summary.append([p, components, noise, force, train_steps, ' '.join(optim_groups), prior, m, s, eval_num, df.success.mean(), model, base])
            else:
                if components not in base_models:
                    print(f'No base model provided for {components} in {args.base_models}')
                    exit(-1)
                summary.append([p, components, noise, force, train_steps, ' '.join(optim_groups), prior, m, s, eval_num, df.success.mean(), model, base, base_models[components]])

    
    pd.DataFrame(summary, columns=header).to_csv(args.out, index=False)
