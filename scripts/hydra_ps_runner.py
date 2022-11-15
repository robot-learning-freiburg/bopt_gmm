import sys

from subprocess import Popen


def power_set(*args):
    if len(args) == 0:
        return []

    if len(args) == 1:
        return [(a, ) for a in args[0]]
    temp = power_set(*args[1:])
    return sum([[(a, ) + t for t in temp] for a in args[0]], [])

def parse_list(list_str, tf=str):
    if list_str[0] != '[' or list_str[-1] != ']':
        raise Exception(f'Expected list string to start with "[" and end with "]"')

    return [tf(i) for i in list_str[1:-1].split(',')]


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
