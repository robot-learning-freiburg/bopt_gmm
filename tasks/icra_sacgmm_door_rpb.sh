python scripts/sacgmm_main.py door_sacgmm --data-dir /home/aroefer/icra_sacgmm_door_rpbuffer/sacgmm_rb_door_p7_n00_10_05_10_wpe_sR_1 --run-prefix icra_rb_sacgmm_door_p7_n00_10_05_10_wpe_sR_1 --deep-eval 20 --ckpt-freq 25 --wandb --replay-buffer /home/aroefer/test_rp_buffer_generation/is_door_p7_n00_hpo_wpe_77ad5a/replay_buffer --overrides bopt_agent.gmm.model=models/gmm/door_7p.npy env.noise.position.variance=0.00 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=eigen +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir /home/aroefer/icra_sacgmm_door_rpbuffer/sacgmm_rb_door_p7_n00_10_05_10_wpe_sR_2 --run-prefix icra_rb_sacgmm_door_p7_n00_10_05_10_wpe_sR_1 --deep-eval 20 --ckpt-freq 25 --wandb --replay-buffer /home/aroefer/test_rp_buffer_generation/is_door_p7_n00_hpo_wpe_77ad5a/replay_buffer --overrides bopt_agent.gmm.model=models/gmm/door_7p.npy env.noise.position.variance=0.00 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=eigen +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir /home/aroefer/icra_sacgmm_door_rpbuffer/sacgmm_rb_door_p7_n00_10_05_10_wpe_sR_3 --run-prefix icra_rb_sacgmm_door_p7_n00_10_05_10_wpe_sR_1 --deep-eval 20 --ckpt-freq 25 --wandb --replay-buffer /home/aroefer/test_rp_buffer_generation/is_door_p7_n00_hpo_wpe_77ad5a/replay_buffer --overrides bopt_agent.gmm.model=models/gmm/door_7p.npy env.noise.position.variance=0.00 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=eigen +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir /home/aroefer/icra_sacgmm_door_rpbuffer/sacgmm_rb_door_p7_n00_10_05_10_wpe_sR_4 --run-prefix icra_rb_sacgmm_door_p7_n00_10_05_10_wpe_sR_1 --deep-eval 20 --ckpt-freq 25 --wandb --replay-buffer /home/aroefer/test_rp_buffer_generation/is_door_p7_n00_hpo_wpe_77ad5a/replay_buffer --overrides bopt_agent.gmm.model=models/gmm/door_7p.npy env.noise.position.variance=0.00 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=eigen +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir /home/aroefer/icra_sacgmm_door_rpbuffer/sacgmm_rb_door_p7_n00_10_05_10_wpe_sR_5 --run-prefix icra_rb_sacgmm_door_p7_n00_10_05_10_wpe_sR_1 --deep-eval 20 --ckpt-freq 25 --wandb --replay-buffer /home/aroefer/test_rp_buffer_generation/is_door_p7_n00_hpo_wpe_77ad5a/replay_buffer --overrides bopt_agent.gmm.model=models/gmm/door_7p.npy env.noise.position.variance=0.00 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=eigen +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir /home/aroefer/icra_sacgmm_door_rpbuffer/sacgmm_rb_door_p7_n00_10_05_10_wpe_sR_6 --run-prefix icra_rb_sacgmm_door_p7_n00_10_05_10_wpe_sR_1 --deep-eval 20 --ckpt-freq 25 --wandb --replay-buffer /home/aroefer/test_rp_buffer_generation/is_door_p7_n00_hpo_wpe_77ad5a/replay_buffer --overrides bopt_agent.gmm.model=models/gmm/door_7p.npy env.noise.position.variance=0.00 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=eigen +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir /home/aroefer/icra_sacgmm_door_rpbuffer/sacgmm_rb_door_p7_n00_10_05_10_wpe_sR_7 --run-prefix icra_rb_sacgmm_door_p7_n00_10_05_10_wpe_sR_1 --deep-eval 20 --ckpt-freq 25 --wandb --replay-buffer /home/aroefer/test_rp_buffer_generation/is_door_p7_n00_hpo_wpe_77ad5a/replay_buffer --overrides bopt_agent.gmm.model=models/gmm/door_7p.npy env.noise.position.variance=0.00 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=eigen +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir /home/aroefer/icra_sacgmm_door_rpbuffer/sacgmm_rb_door_p7_n00_10_05_10_wpe_sR_8 --run-prefix icra_rb_sacgmm_door_p7_n00_10_05_10_wpe_sR_1 --deep-eval 20 --ckpt-freq 25 --wandb --replay-buffer /home/aroefer/test_rp_buffer_generation/is_door_p7_n00_hpo_wpe_77ad5a/replay_buffer --overrides bopt_agent.gmm.model=models/gmm/door_7p.npy env.noise.position.variance=0.00 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=eigen +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir /home/aroefer/icra_sacgmm_door_rpbuffer/sacgmm_rb_door_p7_n00_10_05_10_wpe_sR_9 --run-prefix icra_rb_sacgmm_door_p7_n00_10_05_10_wpe_sR_1 --deep-eval 20 --ckpt-freq 25 --wandb --replay-buffer /home/aroefer/test_rp_buffer_generation/is_door_p7_n00_hpo_wpe_77ad5a/replay_buffer --overrides bopt_agent.gmm.model=models/gmm/door_7p.npy env.noise.position.variance=0.00 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=eigen +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
python scripts/sacgmm_main.py door_sacgmm --data-dir /home/aroefer/icra_sacgmm_door_rpbuffer/sacgmm_rb_door_p7_n00_10_05_10_wpe_sR_10 --run-prefix icra_rb_sacgmm_door_p7_n00_10_05_10_wpe_sR_1 --deep-eval 20 --ckpt-freq 25 --wandb --replay-buffer /home/aroefer/test_rp_buffer_generation/is_door_p7_n00_hpo_wpe_77ad5a/replay_buffer --overrides bopt_agent.gmm.model=models/gmm/door_7p.npy env.noise.position.variance=0.00 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] +bopt_agent.opt_dims.cvars.type=eigen +bopt_agent.opt_dims.cvars.nary=[position|velocity] +bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null sacgmm.observations=[position,force,torque] sacgmm.lr_actor=2e-3 sacgmm.lr_critic=2e-3 sacgmm.lr_alpha=2e-3 sacgmm.batch_size=256
