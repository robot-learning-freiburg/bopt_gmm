import bopt_gmm.gmm as libgmm
import numpy        as np
import yaml         as yaml

from argparse      import ArgumentParser
from bopt_gmm.bopt import GMMOptAgent
from omegaconf     import OmegaConf
from pathlib       import Path
from tqdm          import tqdm


def shift_updates(agent : GMMOptAgent, update, updates):
    new_updates = []
    ref_gmm     = agent.update_model(update, inplace=False)

    for u in tqdm(updates, desc='Shifting Updates'):
        u_new = {k: v - update[k] for k, v in u.items()}
        
        # Offset the post-hoc shift 
        if agent.does_prior_update:
            u_prior,_ = agent._decode_prior_update(u)
            u_gmm     = agent.base_model.update_gaussian(priors=u_prior)
            prior_delta = u_gmm.pi() - ref_gmm.pi()
            new_prior_u = agent._encode_prior_update(prior_delta)
            u_new.update(new_prior_u)
        
        new_updates.append(u_new)

    return new_updates


def main():
    parser = ArgumentParser(description='Bli bla blu, sandbox')
    parser.add_argument('config',   type=Path, help='Yaml config that holds the agent configuration under "bopt_agent".')
    parser.add_argument('base_gmm', type=Path, help='Base GMM to start with.')
    parser.add_argument('configs',  type=Path, help='Path of configs npz.')
    args = parser.parse_args()

    # Necessary to interpret the update dicts
    agent_cfg  = OmegaConf.load(args.config).bopt_agent
    agent_cfg.gmm.model = args.base_gmm
    
    # Load base GMM and GMM updates
    configs  = np.load(args.configs, allow_pickle=True)['arr_0']
    base_gmm = libgmm.load_gmm(agent_cfg.gmm)

    # Update that we use to create our new GMM base
    base_update  = configs[0]
    
    base_agent = GMMOptAgent(base_gmm, agent_cfg)
    
    # Agent used to check new updates
    new_agent  = GMMOptAgent(base_agent.update_model(base_update, inplace=False), agent_cfg)

    updated_configs = shift_updates(base_agent, base_update, configs)

    deltas = []

    for update_o, update_n in tqdm(zip(configs, updated_configs), desc='Computing deltas'):
        gmm_o = base_agent.update_model(update_o, inplace=False)
        gmm_n = new_agent.update_model(update_n, inplace=False)

        delta_weights = np.abs(gmm_n.pi() - gmm_o.pi())
        delta_means   = np.abs(gmm_n.mu() - gmm_o.mu()).max()
        delta_sigma   = np.abs(gmm_n.sigma() - gmm_o.sigma()).max()

        deltas.append(list(delta_weights) + [delta_means, delta_sigma])

    print(np.max(deltas, axis=0))
    print(np.std(deltas, axis=0))


if __name__ == '__main__':
    main()
