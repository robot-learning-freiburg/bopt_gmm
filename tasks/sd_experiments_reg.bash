python scripts/main.py sliding_door_dbopt --data-dir /home/aroefer/bopt_gmm_exp_sd/smac_50_5_n00_10_05_00_hpo --run-prefix ssmac_sd_50_5_n00_10_05_00_hpo --deep-eval 20 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/sd_5p.npy env.noise.position.variance=0.0 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.00 bopt_agent.opt_dims.means=[position,velocity] +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=50
python scripts/main.py sliding_door_dbopt --data-dir /home/aroefer/bopt_gmm_exp_sd/smac_50_5_n00_10_05_00_bb --run-prefix ssmac_sd_50_5_n00_10_05_00_bb --deep-eval 20 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/sd_5p.npy env.noise.position.variance=0.0 bopt_agent.acq_optimizer=bb bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.00 bopt_agent.opt_dims.means=[position,velocity] +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=50
python scripts/main.py sliding_door_dbopt --data-dir /home/aroefer/bopt_gmm_exp_sd/smac_50_5_n00_10_05_00_hb --run-prefix ssmac_sd_50_5_n00_10_05_00_hb --deep-eval 20 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/sd_5p.npy env.noise.position.variance=0.0 bopt_agent.acq_optimizer=hb bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.00 bopt_agent.opt_dims.means=[position,velocity] +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=50