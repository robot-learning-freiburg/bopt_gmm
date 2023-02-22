python scripts/collect_trajectories.py sliding_door_scenario data/sliding_door_3p_n00_trajectories --samples 15 --overrides bopt_agent.gmm.model=models/gmm/sd_3p.npy env.noise.position.variance=0.00
python scripts/collect_trajectories.py sliding_door_scenario data/sliding_door_3p_n01_trajectories --samples 15 --overrides bopt_agent.gmm.model=models/gmm/sd_3p.npy env.noise.position.variance=0.01
python scripts/collect_trajectories.py sliding_door_scenario data/sliding_door_3p_n02_trajectories --samples 15 --overrides bopt_agent.gmm.model=models/gmm/sd_3p.npy env.noise.position.variance=0.02
python scripts/collect_trajectories.py sliding_door_scenario data/sliding_door_3p_n03_trajectories --samples 15 --overrides bopt_agent.gmm.model=models/gmm/sd_3p.npy env.noise.position.variance=0.03
python scripts/collect_trajectories.py sliding_door_scenario data/sliding_door_3p_n05_trajectories --samples 15 --overrides bopt_agent.gmm.model=models/gmm/sd_3p.npy env.noise.position.variance=0.05
python scripts/collect_trajectories.py sliding_door_scenario data/sliding_door_5p_n00_trajectories --samples 15 --overrides bopt_agent.gmm.model=models/gmm/sd_5p.npy env.noise.position.variance=0.00
python scripts/collect_trajectories.py sliding_door_scenario data/sliding_door_5p_n01_trajectories --samples 15 --overrides bopt_agent.gmm.model=models/gmm/sd_5p.npy env.noise.position.variance=0.01
python scripts/collect_trajectories.py sliding_door_scenario data/sliding_door_5p_n02_trajectories --samples 15 --overrides bopt_agent.gmm.model=models/gmm/sd_5p.npy env.noise.position.variance=0.02
python scripts/collect_trajectories.py sliding_door_scenario data/sliding_door_5p_n03_trajectories --samples 15 --overrides bopt_agent.gmm.model=models/gmm/sd_5p.npy env.noise.position.variance=0.03
python scripts/collect_trajectories.py sliding_door_scenario data/sliding_door_5p_n05_trajectories --samples 15 --overrides bopt_agent.gmm.model=models/gmm/sd_5p.npy env.noise.position.variance=0.05
python scripts/collect_trajectories.py sliding_door_scenario data/sliding_door_7p_n00_trajectories --samples 15 --overrides bopt_agent.gmm.model=models/gmm/sd_7p.npy env.noise.position.variance=0.00
python scripts/collect_trajectories.py sliding_door_scenario data/sliding_door_7p_n01_trajectories --samples 15 --overrides bopt_agent.gmm.model=models/gmm/sd_7p.npy env.noise.position.variance=0.01
python scripts/collect_trajectories.py sliding_door_scenario data/sliding_door_7p_n02_trajectories --samples 15 --overrides bopt_agent.gmm.model=models/gmm/sd_7p.npy env.noise.position.variance=0.02
python scripts/collect_trajectories.py sliding_door_scenario data/sliding_door_7p_n03_trajectories --samples 15 --overrides bopt_agent.gmm.model=models/gmm/sd_7p.npy env.noise.position.variance=0.03
python scripts/collect_trajectories.py sliding_door_scenario data/sliding_door_7p_n05_trajectories --samples 15 --overrides bopt_agent.gmm.model=models/gmm/sd_7p.npy env.noise.position.variance=0.05