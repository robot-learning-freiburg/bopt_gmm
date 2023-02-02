import glob
import pandas as pd 

from argparse import ArgumentParser
from pathlib  import Path


if __name__ == '__main__':
    parser = ArgumentParser(description='Summarize eval stats from a collection of folders.')
    parser.add_argument('dirs', nargs='+', help='Directory to process.')
    parser.add_argument('--out', default='summary.csv', help='Name of file containing the summary.')
    args = parser.parse_args()


    header  = 'path eval accuracy'.split(' ')
    summary = []

    for d in args.dirs:
        pattern = f'{d}/eval_*_ic.csv'

        for p in glob.glob(pattern):
            p = Path(p)
            
            try:
                eval_num = int(p.name[5:-7])
            except ValueError as e:
                print(f'Error raised while processing file {p}:\n{e}')
                exit(0)

            df = pd.read_csv(p)
            summary.append([p, eval_num, df.success.mean()])

    
    pd.DataFrame(summary, columns=header).to_csv(args.out, index=False)

