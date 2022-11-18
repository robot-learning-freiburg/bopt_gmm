import sys

from subprocess import Popen

from bopt_gmm.utils import power_set, \
                           parse_list

if __name__ == '__main__':
    args = sys.argv[1:]
    
    for x in range(len(args)):
        if args[x] == '--overrides':
            break
    
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

    processes = [Popen(['python'] + p_args + ['--overrides'] + [f'{k}={a}' for k, a in zip(arg_names, aset)] + fixed_args) for aset in arg_sets]

    print(f'Launched {len(processes)} processes. Arguments are:')
    
    for argset in arg_sets:
        print('\n'.join(f'{a}={v}' for a, v in zip(arg_names, argset)))

    print('Waiting for their completion...')

    for p in processes:
        p.wait()

    print('Done')
