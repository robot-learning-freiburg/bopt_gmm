import hashlib 
import operator
import omegaconf as oc
import numpy     as np

from functools import reduce
from pathlib import Path


def gen_trajectory_from_transitions(transition_trajectories, deltaT):
    data = []
    for t in transition_trajectories:
        out_traj = []
        for x, (pobs, _, _, _, _) in enumerate(t):
            if x == 0: # Skip because we cannot generate vel
                continue
            p = pobs['position']
            v = (p - t[x-1][0]['position']) / deltaT
            out_traj.append(np.hstack((p, v, pobs['force'])))
        data.append(np.vstack(out_traj))
    return data


def conf_checksum(cfg):
    return hashlib.md5(oc.OmegaConf.to_yaml(cfg).encode('utf-8')).hexdigest()[:6]


def flatten_conf(cfg, prefix='', include_path=False):
    out = {}
    if type(cfg) == oc.dictconfig.DictConfig:
        for k, v in cfg:
            pass


def list_of_dicts_to_dict(input: list) -> dict:
    """
    Transform a dict to a list of dicts
    e.g.
        input
            list =[{'a': 0, 'b': 2},
             {'a': 1, 'b': 3}]
        return
            dict = {'a': [0, 1], 'b': [2, 3]}
    """
    # Init dict with empty array
    output = {}
    for key in input[0].keys():
        output[key] = []

    # Append items in dict values
    for item in input:
        for key in item.keys():
            output[key].append(item[key])

    # Dict with np arrays if input contained np arrays
    for key in input[0].keys():
        if isinstance(input[0][key], np.ndarray):
            output[key] = np.array(output[key])

    return output


def save_demo_npz(observations, save_dir):
    dir_path = Path(save_dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    filename = dir_path / ("demo_%03d" % len(list(dir_path.glob("*"))))
    obs_dict = observations
    if isinstance(observations, list):
        obs_dict = list_of_dicts_to_dict(observations)
    np.savez(filename, **obs_dict)
    print(f"Saved file {filename}")
    return filename

def unpack_transition_traj(t, white_list=None):
    iter_order = []
    fields = []
    data   = []
    groups = []
    for (pobs, _, _, _, _) in t:
        if type(pobs) == dict:
            if len(fields) == 0:
                for k, v in pobs.items():
                    if white_list is not None and k not in white_list:
                        continue

                    iter_order.append(k)
                    try:
                        groups.append([len(fields) + x for x in range(len(v))])
                        fields += [f'{k}_{x}' for x in range(len(v))]
                    except TypeError:
                        groups.append([len(fields)])
                        fields.append(k)
            
            data.append(np.hstack([pobs[k] for k in iter_order]))
        else:
            if len(fields) == 0:
                fields = [f'dim_{x}' for x in range(pobs)]
                groups.append(list(range(len(fields))))
    
            data.append(pobs)
    
    return fields, groups, np.vstack(data)


def unpack_trajectories(file_paths, traj_files, white_list=None):
    trajs = []
    for fp, tf in zip(file_paths, traj_files):

        if tf[tf.files[0]].dtype != object:
            t = []
            groups    = []
            dim_names = []
            for f in tf.files:
                if white_list is not None and f not in white_list:
                    continue

                st = tf[f]
                if st.ndim == 1:
                    st = st.reshape((st.shape[0], 1))
                elif st.ndim > 2:
                    st = st.reshape((st.shape[0], (reduce(operator.mul, st.shape[1:], 1))))
                groups.append([len(dim_names) + x for x in range(st.shape[1])])
                dim_names += [f'{f}_{x}' for x in range(st.shape[1])]
                t.append(st)
                
            trajs.append((fp, dim_names, groups, np.hstack(t)))
        else:
            for x, t in enumerate(tf[tf.files[0]]):
                trajs.append((f'{fp} {x}', ) + unpack_transition_traj(t, white_list))
    return trajs


def gauss_smoothing(o_series, steps):
    if len(o_series.shape) < 2:
        series = o_series.reshape((o_series.shape[0], 1))
    else:
        series = o_series

    acc = []
    smooth_op = np.array([(1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)  for x in np.linspace(-2, 2, steps * 2 + 1)])

    # Slow :(
    for s in series.T:
        m_series =  np.zeros((len(s), steps * 2 + 1))
        m_series[:steps]   = s[0]
        m_series[-steps:]  = s[-1]
        m_series[:, steps] = s

        for x in range(1, steps + 1):
            m_series[ :-x, steps + x] = s[x:]
            m_series[x:  , steps - x] = s[:-x]

        acc.append(m_series @ smooth_op)
    return np.vstack(acc).T.reshape(o_series.shape)


def normalize_trajectories(trajectories, norm_group : str):
    fp, dim_names, groups, data = trajectories[0]
    for dx, dn in enumerate(dim_names):
        if dn[:len(norm_group)] == norm_group:
            break
    else:
        raise Exception(f'Could not find a dimension with prefix {norm_group}')
    
    full_data = np.vstack([data for _, _, _, data in trajectories])

    for gx, g in enumerate(groups):
        if dx in g:
            sub_data = np.take(full_data, g, axis=1)
            norm_lin_span = np.sqrt(np.sum((sub_data.max(axis=1) - sub_data.min(axis=1))**2))
            break

    group_factors = []
    group_names   = []
    for g in groups:
        sub_data = np.take(full_data, g, axis=1)
        lin_span = np.sqrt(np.sum((sub_data.max(axis=1) - sub_data.min(axis=1))**2))
        group_factors.append(norm_lin_span / lin_span)
        dim_name = dim_names[g[0]]
        dim_name = dim_name[:dim_name.rfind('_')]
        group_names.append(dim_name)

    out = []
    for fp, dim_names, groups, data in trajectories:
        out_data = np.hstack(np.take(data, g, axis=1) * gf for g, gf in zip(groups, group_factors))
        out.append((fp, dim_names, groups, out_data))

    return out, dict(zip(group_names, group_factors))


def calculate_trajectory_velocities(t : np.ndarray, delta_t):
    return np.hstack((t[1:], (t[1:] - t[:-1]) / delta_t))


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
    import hydra

    hydra.initialize(config_path="../../config")
    cfg = hydra.compose('bopt_gmm_peg_env_fbopt')

    print(type(cfg))
