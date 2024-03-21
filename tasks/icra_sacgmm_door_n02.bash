# Just means
python scripts/sacgmm_main.py door_sacgmm --data-dir results/icra_sacgmm_exp_door/is_door_p5_n02_10_05_10_wp_sR_1 --run-prefix icra_is_sacgmm_door_p5_n02_10_05_10_wp_sR --deep-eval 20 --ckpt-freq 25 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.02 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.0 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=eigen +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir results/icra_sacgmm_exp_door/is_door_p5_n02_10_05_10_wp_sR_2 --run-prefix icra_is_sacgmm_door_p5_n02_10_05_10_wp_sR --deep-eval 20 --ckpt-freq 25 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.02 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.0 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=eigen +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir results/icra_sacgmm_exp_door/is_door_p5_n02_10_05_10_wp_sR_3 --run-prefix icra_is_sacgmm_door_p5_n02_10_05_10_wp_sR --deep-eval 20 --ckpt-freq 25 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.02 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.0 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=eigen +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir results/icra_sacgmm_exp_door/is_door_p5_n02_10_05_10_wp_sR_4 --run-prefix icra_is_sacgmm_door_p5_n02_10_05_10_wp_sR --deep-eval 20 --ckpt-freq 25 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.02 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.0 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=eigen +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir results/icra_sacgmm_exp_door/is_door_p5_n02_10_05_10_wp_sR_5 --run-prefix icra_is_sacgmm_door_p5_n02_10_05_10_wp_sR --deep-eval 20 --ckpt-freq 25 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.02 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.0 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=eigen +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir results/icra_sacgmm_exp_door/is_door_p5_n02_10_05_10_wp_sR_6 --run-prefix icra_is_sacgmm_door_p5_n02_10_05_10_wp_sR --deep-eval 20 --ckpt-freq 25 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.02 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.0 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=eigen +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256

# Just covar
python scripts/sacgmm_main.py door_sacgmm --data-dir results/icra_sacgmm_exp_door/is_door_p5_n02_10_05_10_we_sR_1 --run-prefix icra_is_sacgmm_door_p5_n02_10_05_10_we_sR --deep-eval 20 --ckpt-freq 25 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.02 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.0 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=eigen +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir results/icra_sacgmm_exp_door/is_door_p5_n02_10_05_10_we_sR_2 --run-prefix icra_is_sacgmm_door_p5_n02_10_05_10_we_sR --deep-eval 20 --ckpt-freq 25 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.02 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.0 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=eigen +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir results/icra_sacgmm_exp_door/is_door_p5_n02_10_05_10_we_sR_3 --run-prefix icra_is_sacgmm_door_p5_n02_10_05_10_we_sR --deep-eval 20 --ckpt-freq 25 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.02 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.0 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=eigen +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir results/icra_sacgmm_exp_door/is_door_p5_n02_10_05_10_we_sR_4 --run-prefix icra_is_sacgmm_door_p5_n02_10_05_10_we_sR --deep-eval 20 --ckpt-freq 25 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.02 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.0 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=eigen +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir results/icra_sacgmm_exp_door/is_door_p5_n02_10_05_10_we_sR_5 --run-prefix icra_is_sacgmm_door_p5_n02_10_05_10_we_sR --deep-eval 20 --ckpt-freq 25 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.02 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.0 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=eigen +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir results/icra_sacgmm_exp_door/is_door_p5_n02_10_05_10_we_sR_6 --run-prefix icra_is_sacgmm_door_p5_n02_10_05_10_we_sR --deep-eval 20 --ckpt-freq 25 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.02 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.0 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=eigen +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256

python scripts/sacgmm_main.py door_sacgmm --data-dir results/icra_sacgmm_exp_door/is_door_p5_n02_10_05_10_wc_sR_1 --run-prefix icra_is_sacgmm_door_p5_n02_10_05_10_wc_sR --deep-eval 20 --ckpt-freq 25 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.02 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.0 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=rotation +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir results/icra_sacgmm_exp_door/is_door_p5_n02_10_05_10_wc_sR_2 --run-prefix icra_is_sacgmm_door_p5_n02_10_05_10_wc_sR --deep-eval 20 --ckpt-freq 25 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.02 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.0 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=rotation +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir results/icra_sacgmm_exp_door/is_door_p5_n02_10_05_10_wc_sR_3 --run-prefix icra_is_sacgmm_door_p5_n02_10_05_10_wc_sR --deep-eval 20 --ckpt-freq 25 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.02 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.0 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=rotation +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir results/icra_sacgmm_exp_door/is_door_p5_n02_10_05_10_wc_sR_4 --run-prefix icra_is_sacgmm_door_p5_n02_10_05_10_wc_sR --deep-eval 20 --ckpt-freq 25 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.02 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.0 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=rotation +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir results/icra_sacgmm_exp_door/is_door_p5_n02_10_05_10_wc_sR_5 --run-prefix icra_is_sacgmm_door_p5_n02_10_05_10_wc_sR --deep-eval 20 --ckpt-freq 25 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.02 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.0 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=rotation +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir results/icra_sacgmm_exp_door/is_door_p5_n02_10_05_10_wc_sR_6 --run-prefix icra_is_sacgmm_door_p5_n02_10_05_10_wc_sR --deep-eval 20 --ckpt-freq 25 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.02 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.0 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=rotation +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256

# Means and covar
python scripts/sacgmm_main.py door_sacgmm --data-dir results/icra_sacgmm_exp_door/is_door_p5_n02_10_05_10_wpe_sR_1 --run-prefix icra_is_sacgmm_door_p5_n02_10_05_10_wpe_sR --deep-eval 20 --ckpt-freq 25 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.02 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=eigen +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir results/icra_sacgmm_exp_door/is_door_p5_n02_10_05_10_wpe_sR_2 --run-prefix icra_is_sacgmm_door_p5_n02_10_05_10_wpe_sR --deep-eval 20 --ckpt-freq 25 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.02 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=eigen +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir results/icra_sacgmm_exp_door/is_door_p5_n02_10_05_10_wpe_sR_3 --run-prefix icra_is_sacgmm_door_p5_n02_10_05_10_wpe_sR --deep-eval 20 --ckpt-freq 25 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.02 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=eigen +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir results/icra_sacgmm_exp_door/is_door_p5_n02_10_05_10_wpe_sR_4 --run-prefix icra_is_sacgmm_door_p5_n02_10_05_10_wpe_sR --deep-eval 20 --ckpt-freq 25 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.02 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=eigen +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir results/icra_sacgmm_exp_door/is_door_p5_n02_10_05_10_wpe_sR_5 --run-prefix icra_is_sacgmm_door_p5_n02_10_05_10_wpe_sR --deep-eval 20 --ckpt-freq 25 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.02 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=eigen +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir results/icra_sacgmm_exp_door/is_door_p5_n02_10_05_10_wpe_sR_6 --run-prefix icra_is_sacgmm_door_p5_n02_10_05_10_wpe_sR --deep-eval 20 --ckpt-freq 25 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.02 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=eigen +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256

python scripts/sacgmm_main.py door_sacgmm --data-dir results/icra_sacgmm_exp_door/is_door_p5_n02_10_05_10_wpc_sR_1 --run-prefix icra_is_sacgmm_door_p5_n02_10_05_10_wpc_sR --deep-eval 20 --ckpt-freq 25 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.02 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=rotation +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir results/icra_sacgmm_exp_door/is_door_p5_n02_10_05_10_wpc_sR_2 --run-prefix icra_is_sacgmm_door_p5_n02_10_05_10_wpc_sR --deep-eval 20 --ckpt-freq 25 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.02 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=rotation +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir results/icra_sacgmm_exp_door/is_door_p5_n02_10_05_10_wpc_sR_3 --run-prefix icra_is_sacgmm_door_p5_n02_10_05_10_wpc_sR --deep-eval 20 --ckpt-freq 25 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.02 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=rotation +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir results/icra_sacgmm_exp_door/is_door_p5_n02_10_05_10_wpc_sR_4 --run-prefix icra_is_sacgmm_door_p5_n02_10_05_10_wpc_sR --deep-eval 20 --ckpt-freq 25 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.02 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=rotation +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir results/icra_sacgmm_exp_door/is_door_p5_n02_10_05_10_wpc_sR_5 --run-prefix icra_is_sacgmm_door_p5_n02_10_05_10_wpc_sR --deep-eval 20 --ckpt-freq 25 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.02 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=rotation +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir results/icra_sacgmm_exp_door/is_door_p5_n02_10_05_10_wpc_sR_6 --run-prefix icra_is_sacgmm_door_p5_n02_10_05_10_wpc_sR --deep-eval 20 --ckpt-freq 25 --wandb --overrides bopt_agent.gmm.model=models/gmm/door_5p.npy env.noise.position.variance=0.02 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=rotation +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
