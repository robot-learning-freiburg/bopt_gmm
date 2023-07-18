import sys

from pathlib import Path

from bopt_gmm.utils import JobRunner

if __name__ == '__main__':
    args = sys.argv[1:]

    try:
        nproc = int(args[0])
        args = args[1:]
    except ValueError as e:
        nproc = 10

    jobs = []
    for a in args:
        if Path(a).exists() and Path(a).is_file():
            with open(a, 'r') as f:
                for l in f.readlines():
                    l = l.strip()
                    if len(l) > 0 and l[0] != '#':
                        jobs.append(l)
        else:
            jobs.append(a)

    runner = JobRunner([[a for a in j.strip().split(' ') if a != ''] for j in jobs], nproc)
    print(f'Running {len(jobs)}...')
    runner.run()
    print('Done')
