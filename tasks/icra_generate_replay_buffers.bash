# # --- DOOR ---
# # N 0.00
# # Just mean
# python scripts/main.py door_dbopt_cvar --data-dir results/icra_replay_buffers/door_p5_n00_10_05_10_wp_sR --run-prefix icra_rb_door_p5_n00_10_05_10_wp_sR --deep-eval 20 --sacgmm-steps 30 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.00 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.0 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=eigen bopt_agent.opt_dims.cvars.nary=[position|velocity] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=50 bopt_agent.incumbent_eval=null
# # Just Covar
# python scripts/main.py door_dbopt_cvar --data-dir results/icra_replay_buffers/door_p5_n00_10_05_10_we_sR --run-prefix icra_rb_door_p5_n00_10_05_10_we_sR --deep-eval 20 --sacgmm-steps 30 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.00 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.0 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=eigen bopt_agent.opt_dims.cvars.nary=[position|velocity] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=50 bopt_agent.incumbent_eval=null
# python scripts/main.py door_dbopt_cvar --data-dir results/icra_replay_buffers/door_p5_n00_10_05_10_wc_sR --run-prefix icra_rb_door_p5_n00_10_05_10_wc_sR --deep-eval 20 --sacgmm-steps 30 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.00 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.0 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=rotation bopt_agent.opt_dims.cvars.nary=[position|velocity] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=50 bopt_agent.incumbent_eval=null
# # All
# python scripts/main.py door_dbopt_cvar --data-dir results/icra_replay_buffers/door_p5_n00_10_05_10_wpe_sR --run-prefix icra_rb_door_p5_n00_10_05_10_wpe_sR --deep-eval 20 --sacgmm-steps 30 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.00 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=eigen bopt_agent.opt_dims.cvars.nary=[position|velocity] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=50 bopt_agent.incumbent_eval=null
# python scripts/main.py door_dbopt_cvar --data-dir results/icra_replay_buffers/door_p5_n00_10_05_10_wpc_sR --run-prefix icra_rb_door_p5_n00_10_05_10_wpc_sR --deep-eval 20 --sacgmm-steps 30 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.00 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=rotation bopt_agent.opt_dims.cvars.nary=[position|velocity] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=50 bopt_agent.incumbent_eval=null

# N 0.02
# Just mean
python scripts/main.py door_dbopt_cvar --data-dir results/icra_replay_buffers/door_p5_n01_10_05_10_wp_sR --run-prefix icra_rb_door_p5_n01_10_05_10_wp_sR --deep-eval 20 --sacgmm-steps 30 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.0 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=eigen bopt_agent.opt_dims.cvars.nary=[position|velocity] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=50 bopt_agent.incumbent_eval=null
# Just Covar
python scripts/main.py door_dbopt_cvar --data-dir results/icra_replay_buffers/door_p5_n01_10_05_10_we_sR --run-prefix icra_rb_door_p5_n01_10_05_10_we_sR --deep-eval 20 --sacgmm-steps 30 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.0 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=eigen bopt_agent.opt_dims.cvars.nary=[position|velocity] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=50 bopt_agent.incumbent_eval=null
python scripts/main.py door_dbopt_cvar --data-dir results/icra_replay_buffers/door_p5_n01_10_05_10_wc_sR --run-prefix icra_rb_door_p5_n01_10_05_10_wc_sR --deep-eval 20 --sacgmm-steps 30 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.0 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=rotation bopt_agent.opt_dims.cvars.nary=[position|velocity] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=50 bopt_agent.incumbent_eval=null
# All
python scripts/main.py door_dbopt_cvar --data-dir results/icra_replay_buffers/door_p5_n01_10_05_10_wpe_sR --run-prefix icra_rb_door_p5_n01_10_05_10_wpe_sR --deep-eval 20 --sacgmm-steps 30 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=eigen bopt_agent.opt_dims.cvars.nary=[position|velocity] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=50 bopt_agent.incumbent_eval=null
python scripts/main.py door_dbopt_cvar --data-dir results/icra_replay_buffers/door_p5_n01_10_05_10_wpc_sR --run-prefix icra_rb_door_p5_n01_10_05_10_wpc_sR --deep-eval 20 --sacgmm-steps 30 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=rotation bopt_agent.opt_dims.cvars.nary=[position|velocity] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=50 bopt_agent.incumbent_eval=null

# # -- HATCH ---
# # N 0.00
# # Just mean
# python scripts/main.py sliding_door_dbopt_cvar --data-dir results/icra_replay_buffers/sd_p5_n00_10_05_10_wp_sR --run-prefix icra_rb_sd_p5_n00_10_05_10_wp_sR --deep-eval 20 --sacgmm-steps 30 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/sd_5p.npy env.noise.position.variance=0.00 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.0 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=eigen bopt_agent.opt_dims.cvars.nary=[position|velocity] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=50 bopt_agent.incumbent_eval=null
# # Just covar
# python scripts/main.py sliding_door_dbopt_cvar --data-dir results/icra_replay_buffers/sd_p5_n00_10_05_10_we_sR --run-prefix icra_rb_sd_p5_n00_10_05_10_we_sR --deep-eval 20 --sacgmm-steps 30 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/sd_5p.npy env.noise.position.variance=0.00 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.0 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=eigen bopt_agent.opt_dims.cvars.nary=[position|velocity] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=50 bopt_agent.incumbent_eval=null
# python scripts/main.py sliding_door_dbopt_cvar --data-dir results/icra_replay_buffers/sd_p5_n00_10_05_10_wc_sR --run-prefix icra_rb_sd_p5_n00_10_05_10_wc_sR --deep-eval 20 --sacgmm-steps 30 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/sd_5p.npy env.noise.position.variance=0.00 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.0 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=rotation bopt_agent.opt_dims.cvars.nary=[position|velocity] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=50 bopt_agent.incumbent_eval=null
# # All 
# python scripts/main.py sliding_door_dbopt_cvar --data-dir results/icra_replay_buffers/sd_p5_n00_10_05_10_wpe_sR --run-prefix icra_rb_sd_p5_n00_10_05_10_wpe_sR --deep-eval 20 --sacgmm-steps 30 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/sd_5p.npy env.noise.position.variance=0.0 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=eigen bopt_agent.opt_dims.cvars.nary=[position|velocity] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=50 bopt_agent.incumbent_eval=null
# python scripts/main.py sliding_door_dbopt_cvar --data-dir results/icra_replay_buffers/sd_p5_n00_10_05_10_wpc_sR --run-prefix icra_rb_sd_p5_n00_10_05_10_wpc_sR --deep-eval 20 --sacgmm-steps 30 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/sd_5p.npy env.noise.position.variance=0.0 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=rotation bopt_agent.opt_dims.cvars.nary=[position|velocity] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=50 bopt_agent.incumbent_eval=null

# N 0.02
# Just means
python scripts/main.py sliding_door_dbopt_cvar --data-dir results/icra_replay_buffers/sd_p5_n01_10_05_10_wp_sR --run-prefix icra_rb_sd_p5_n01_10_05_10_wp_sR --deep-eval 20 --sacgmm-steps 30 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/sd_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.0 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=eigen bopt_agent.opt_dims.cvars.nary=[position|velocity] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=50 bopt_agent.incumbent_eval=null
# Just covar
python scripts/main.py sliding_door_dbopt_cvar --data-dir results/icra_replay_buffers/sd_p5_n01_10_05_10_we_sR --run-prefix icra_rb_sd_p5_n01_10_05_10_we_sR --deep-eval 20 --sacgmm-steps 30 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/sd_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.0 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=eigen bopt_agent.opt_dims.cvars.nary=[position|velocity] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=50 bopt_agent.incumbent_eval=null
python scripts/main.py sliding_door_dbopt_cvar --data-dir results/icra_replay_buffers/sd_p5_n01_10_05_10_wc_sR --run-prefix icra_rb_sd_p5_n01_10_05_10_wc_sR --deep-eval 20 --sacgmm-steps 30 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/sd_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.0 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=rotation bopt_agent.opt_dims.cvars.nary=[position|velocity] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=50 bopt_agent.incumbent_eval=null
# Means and covar
python scripts/main.py sliding_door_dbopt_cvar --data-dir results/icra_replay_buffers/sd_p5_n01_10_05_10_wpe_sR --run-prefix icra_rb_sd_p5_n01_10_05_10_wpe_sR --deep-eval 20 --sacgmm-steps 30 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/sd_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=eigen bopt_agent.opt_dims.cvars.nary=[position|velocity] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=50 bopt_agent.incumbent_eval=null
python scripts/main.py sliding_door_dbopt_cvar --data-dir results/icra_replay_buffers/sd_p5_n01_10_05_10_wpc_sR --run-prefix icra_rb_sd_p5_n01_10_05_10_wpc_sR --deep-eval 20 --sacgmm-steps 30 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/sd_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=rotation bopt_agent.opt_dims.cvars.nary=[position|velocity] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=50 bopt_agent.incumbent_eval=null