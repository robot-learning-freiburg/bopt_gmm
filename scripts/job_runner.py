import sys

from bopt_gmm.utils import JobRunner

if __name__ == '__main__':
    jobs = sys.argv[1:]

    try:
        nproc = int(jobs[0])
        jobs = jobs[1:]
    except ValueError as e:
        nproc = 10

    runner = JobRunner([j.split(' ') for j in jobs], nproc)
    print(f'Running {len(jobs)}...')
    runner.run()
    print('Done')
