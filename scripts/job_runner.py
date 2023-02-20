import sys

from pathlib import Path

from bopt_gmm.utils import JobRunner

if __name__ == '__main__':
    jobs = sys.argv[1:]

    try:
        nproc = int(jobs[0])
        jobs = jobs[1:]
    except ValueError as e:
        nproc = 10

    if Path(jobs[0]).exists() and Path(jobs[0]).is_file():
        with open(jobs[0], 'r') as f:
            jobs = f.readlines()

    runner = JobRunner([[a for a in j.strip().split(' ') if a != ''] for j in jobs], nproc)
    print(f'Running {len(jobs)}...')
    runner.run()
    print('Done')
