import hashlib 
import omegaconf as oc
import numpy     as np

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


if __name__ == '__main__':
    import hydra

    hydra.initialize(config_path="../../config")
    cfg = hydra.compose('bopt_gmm_peg_env_fbopt')

    print(type(cfg))
