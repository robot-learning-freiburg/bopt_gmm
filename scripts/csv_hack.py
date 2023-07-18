import pandas as pd

from argparse import ArgumentParser
from pathlib  import Path

if __name__ == '__main__':
    parser = ArgumentParser(description='Joins a bunch of CSV files in one.')
    parser.add_argument('files', nargs='+', help='CSV files to join.')
    args = parser.parse_args()

    data = [(int(Path(f).name[5:-7]),  pd.read_csv(f).accuracy.mean()) for f in args.files]
    df   = pd.DataFrame(data, columns=['episode', 'accuracy'])

    for i in sorted(df.episode.unique()):
        mean = df[df.episode == i].accuracy.mean()
        print(i, mean, sum(df.episode == i))
