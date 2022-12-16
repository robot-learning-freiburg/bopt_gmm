import sys
import os
import time
import signal 

from subprocess import Popen

from bopt_gmm.utils import power_set, \
                           parse_list

if __name__ == '__main__':
    args = sys.argv[1:]
    
    try:
        i_nproc = args.index('--n-proc') + 1
        nproc   = int(args[i_nproc])
        args    = args[i_nproc + 1:]
    except ValueError:
        nproc = 100

    x = args.index('--overrides')
    
    p_args    = args[:x]
    overrides = args[x+1:]

    fixed_args = {}
    ps_args    = {}

    for oarg in overrides:
        k, v = oarg.split('=')
        if v[0] == '[':
            ps_args[k] = parse_list(v)
        else:
            ps_args[k] = [v]

    arg_names, values = zip(*list(ps_args.items()))

    arg_sets = power_set(*values)
    print(arg_sets)

    fixed_args = [f'{k}={v}' for k, v in fixed_args.items()]

    tasks = [(['python'] + p_args + ['--overrides'] + [f'{k}={a}' for k, a in zip(arg_names, aset)] + fixed_args) for aset in arg_sets]
    
    processes = []

    def sig_handler(sig, *args):
        print('Received SIGINT. Killing subprocesses...')
        for p in processes:
            p.send_signal(signal.SIGINT)
        
        for p in processes:
            p.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)

    with open(os.devnull, 'w') as devnull:
        while len(tasks) > 0:
            while len(processes) < nproc and len(tasks) > 0:
                processes.append(Popen(tasks[0], stdout=devnull))
                print(f'Launched task "{" ".join(tasks[0])}"\nRemaining {len(tasks) - 1}')
                tasks = tasks[1:]

            time.sleep(1.0)

            tidx = 0
            while tidx < len(processes):
                if processes[tidx].poll() is not None:
                    del processes[tidx]
                else:
                    tidx += 1

        print('Waiting for their completion...')
        for p in processes:
            p.wait()
    print('Done')
