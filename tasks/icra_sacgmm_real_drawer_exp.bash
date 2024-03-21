python scripts/sacgmm_main.py real_drawer_sacgmm --show-gui --data-dir results/icra_sacgmm_exp_rdrawer_super_narrow_shorter/is_rdrawer_p3_hpo_wp_sR_2 --run-prefix sacgmm_rdrawer_p3_n00_10_05_10_wp_sR_2 --deep-eval 10 --ckpt-freq 25 --wandb --overrides bopt_agent.num_episode_steps=700 bopt_agent.gmm.model=models/gmm/real_drawer_3p.npy bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.05 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.0 bopt_agent.opt_dims.means=[position] bopt_agent.num_training_cycles=100 sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
