# BOPT-GMM

Implementation of the idea of optimizing GMM motion models using Bayesian Optimization

## Installation

The package can be installed with or without a ROS environment. ROS is required for real experiments and some visualizations.

```bash
# Optional: Create new ROS(-VENV)
createROSWS ws
cd src

# Dependencies
# rl_tasks
https://github.com/robot-learning-freiburg/rl_tasks

# In case of ROS/Catkin
git clone https://github.com/ARoefer/roebots.git
cd roebots
pip install -r requirements.txt
cd ..

# Clone repository
git clone https://github.com/robot-learning-freiburg/bopt_gmm.git
cd bopt_gmm

# If ROSVENV
pip install -r requirements.txt
# If plain pip/conda
pip install -e .

# SMAC always installs the wrong version of ConfigSpace
pip install ConfigSpace==0.6.0

# If ROSVENV
cd ../..
catkin build
reloadROS
```

## Running Demos

All runnable files are located in the `scripts` directory. For BOPT, the main executable is `main.py`. For running SACGMM, the main file is `sacgmm_main.py`. Both files have the same arguments:

```bash
# At bopt_gmm package root
# Runs BOPT-GMM in the sliding door scenario
python scripts/main.py sliding_door_dbopt 

# Runs BOPT-GMM in the sliding door scenario and logs to W&B. All runs are prefixed with "lol_"
python scripts/main.py sliding_door_dbopt --wandb --run-prefix lol_

# Runs BOPT-GMM and logs everything to directories in "~/bopt_experiment" prefixed as "lol_"
python scripts/main.py sliding_door_dbopt --data-dir ~/bopt_experiment/lol_

# Runs BOPT-GMM, saves every incumbent and an evaluates it for 40 episodes
python scripts/main.py sliding_door_dbopt --ckpt-freq 1 --deep-eval 40

# Runs BOPT-GMM, changes the base GMM-model to "models/gmm/sd_3p.npy"
python scripts/main.py sliding_door_dbopt --overrides bopt_agent.gmm.model=models/gmm/sd_3p.npy

# Evaluates GMM
python scripts/main.py sliding_door_dbopt --mode eval-gmm

# Running SAC-GMM in sliding door scenario
python scripts/sacgmm_main.py sliding_door_sacgmm
```

All possible configurations can be found in `config`. Some of the scenarios are older and might not work anymore. The scenarios `door` and `sliding_door` are the most active ones right now.

## Package Structure

The implementation of all components can be found in `src/bopt_gmm`. The subdirectories contain the following:

 - `gmm` Implementation of GMM, including semantic versions stored in `instances.py`. The subdirectory `generation` stores different optimizers for generating GMMs.
 - `envs` Holds the implementations of all environments. These are derived from `gym.Env`. In addition to the typical functions required by `Gym`, the environments also have a `config_space` attribute, and a `config_dict` function which encode the configuration space, and the current configuration of the environment.
 - `baselines` implements various baselines. SAC-GMM is complicated enough for it to get its own directory. To see how SACGMM works in this implementation, please check the `README`in that package.

