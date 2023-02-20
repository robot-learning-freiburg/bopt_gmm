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


    header  = 'path components noise eval accuracy model base'.split(' ')
    if args.base_models is not None:
        header += ['demo_base']
    
    summary = []

    for d in tqdm(args.dirs, desc='Processing directories'):
        pattern = f'{d}/eval_*_ic.csv'

        for p in glob.glob(pattern):
            p = Path(p)
            
            try:
                eval_num = int(p.name[5:-7])
            except ValueError as e:
                print(f'Error raised while processing file {p}:\n{e}')
                exit(0)

            df = pd.read_csv(p)

            noise = int(regex.findall(r"_n\d\d_", str(p))[0][2:-1]) * 0.01
            components = int(regex.findall(r"_\d+_", str(p))[0][1:-1])

            # noise_vectors = np.vstack((df.position_noise_x.array, 
            #                            df.position_noise_y.array,
            #                            df.position_noise_z.array)).T
            # noise = np.sqrt((noise_vectors ** 2).sum(axis=1)).mean()

            model = f'{p.parent}/models/gmm_{eval_num}.npy'
            base  = f'{p.parent}/models/gmm_base.npy'

            if args.base_models is None:
                summary.append([p, components, noise, eval_num, df.success.mean(), model, base])
            else:
                if components not in base_models:
                    print(f'No base model provided for {components} in {args.base_models}')
                    exit(-1)
                summary.append([p, components, noise, eval_num, df.success.mean(), model, base, base_models[components]])

    
    pd.DataFrame(summary, columns=header).to_csv(args.out, index=False)
