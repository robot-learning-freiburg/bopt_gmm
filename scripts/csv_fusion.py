import pandas as pd

from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser(description='Joins a bunch of CSV files in one.')
    parser.add_argument('out', help='New file to write to')
    parser.add_argument('files', nargs='+', help='CSV files to join.')
    args = parser.parse_args()

    df = pd.concat([pd.read_csv(f) for f in args.files], ignore_index=True)
    df.to_csv(args.out, index=False)
