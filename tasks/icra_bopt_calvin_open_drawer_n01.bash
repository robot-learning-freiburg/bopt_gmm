# Just means
python scripts/main.py calvin_open_drawer_dbopt_cvar --data-dir results/icra_bopt_exp_cod/is_cod_p5_n01_10_05_10_wp_sR_1 --run-prefix icra_is_bopt_cod_p5_n01_10_05_10_wp_sR --deep-eval 50 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/calvin_open_drawer_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.0 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=eigen bopt_agent.opt_dims.cvars.nary=['position|velocity'] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null
python scripts/main.py calvin_open_drawer_dbopt_cvar --data-dir results/icra_bopt_exp_cod/is_cod_p5_n01_10_05_10_wp_sR_2 --run-prefix icra_is_bopt_cod_p5_n01_10_05_10_wp_sR --deep-eval 50 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/calvin_open_drawer_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.0 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=eigen bopt_agent.opt_dims.cvars.nary=['position|velocity'] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null
python scripts/main.py calvin_open_drawer_dbopt_cvar --data-dir results/icra_bopt_exp_cod/is_cod_p5_n01_10_05_10_wp_sR_3 --run-prefix icra_is_bopt_cod_p5_n01_10_05_10_wp_sR --deep-eval 50 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/calvin_open_drawer_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.0 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=eigen bopt_agent.opt_dims.cvars.nary=['position|velocity'] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null
python scripts/main.py calvin_open_drawer_dbopt_cvar --data-dir results/icra_bopt_exp_cod/is_cod_p5_n01_10_05_10_wp_sR_4 --run-prefix icra_is_bopt_cod_p5_n01_10_05_10_wp_sR --deep-eval 50 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/calvin_open_drawer_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.0 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=eigen bopt_agent.opt_dims.cvars.nary=['position|velocity'] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null
python scripts/main.py calvin_open_drawer_dbopt_cvar --data-dir results/icra_bopt_exp_cod/is_cod_p5_n01_10_05_10_wp_sR_5 --run-prefix icra_is_bopt_cod_p5_n01_10_05_10_wp_sR --deep-eval 50 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/calvin_open_drawer_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.0 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=eigen bopt_agent.opt_dims.cvars.nary=['position|velocity'] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null
python scripts/main.py calvin_open_drawer_dbopt_cvar --data-dir results/icra_bopt_exp_cod/is_cod_p5_n01_10_05_10_wp_sR_6 --run-prefix icra_is_bopt_cod_p5_n01_10_05_10_wp_sR --deep-eval 50 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/calvin_open_drawer_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.0 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=eigen bopt_agent.opt_dims.cvars.nary=['position|velocity'] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null

# Just covar
python scripts/main.py calvin_open_drawer_dbopt_cvar --data-dir results/icra_bopt_exp_cod/is_cod_p5_n01_10_05_10_we_sR_1 --run-prefix icra_is_bopt_cod_p5_n01_10_05_10_we_sR --deep-eval 50 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/calvin_open_drawer_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.0 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=eigen bopt_agent.opt_dims.cvars.nary=['position|velocity'] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null
python scripts/main.py calvin_open_drawer_dbopt_cvar --data-dir results/icra_bopt_exp_cod/is_cod_p5_n01_10_05_10_we_sR_2 --run-prefix icra_is_bopt_cod_p5_n01_10_05_10_we_sR --deep-eval 50 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/calvin_open_drawer_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.0 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=eigen bopt_agent.opt_dims.cvars.nary=['position|velocity'] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null
python scripts/main.py calvin_open_drawer_dbopt_cvar --data-dir results/icra_bopt_exp_cod/is_cod_p5_n01_10_05_10_we_sR_3 --run-prefix icra_is_bopt_cod_p5_n01_10_05_10_we_sR --deep-eval 50 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/calvin_open_drawer_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.0 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=eigen bopt_agent.opt_dims.cvars.nary=['position|velocity'] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null
python scripts/main.py calvin_open_drawer_dbopt_cvar --data-dir results/icra_bopt_exp_cod/is_cod_p5_n01_10_05_10_we_sR_4 --run-prefix icra_is_bopt_cod_p5_n01_10_05_10_we_sR --deep-eval 50 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/calvin_open_drawer_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.0 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=eigen bopt_agent.opt_dims.cvars.nary=['position|velocity'] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null
python scripts/main.py calvin_open_drawer_dbopt_cvar --data-dir results/icra_bopt_exp_cod/is_cod_p5_n01_10_05_10_we_sR_5 --run-prefix icra_is_bopt_cod_p5_n01_10_05_10_we_sR --deep-eval 50 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/calvin_open_drawer_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.0 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=eigen bopt_agent.opt_dims.cvars.nary=['position|velocity'] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null
python scripts/main.py calvin_open_drawer_dbopt_cvar --data-dir results/icra_bopt_exp_cod/is_cod_p5_n01_10_05_10_we_sR_6 --run-prefix icra_is_bopt_cod_p5_n01_10_05_10_we_sR --deep-eval 50 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/calvin_open_drawer_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.0 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=eigen bopt_agent.opt_dims.cvars.nary=['position|velocity'] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null

python scripts/main.py calvin_open_drawer_dbopt_cvar --data-dir results/icra_bopt_exp_cod/is_cod_p5_n01_10_05_10_wc_sR_1 --run-prefix icra_is_bopt_cod_p5_n01_10_05_10_wc_sR --deep-eval 50 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/calvin_open_drawer_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.0 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=rotation bopt_agent.opt_dims.cvars.nary=['position|velocity'] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null
python scripts/main.py calvin_open_drawer_dbopt_cvar --data-dir results/icra_bopt_exp_cod/is_cod_p5_n01_10_05_10_wc_sR_2 --run-prefix icra_is_bopt_cod_p5_n01_10_05_10_wc_sR --deep-eval 50 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/calvin_open_drawer_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.0 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=rotation bopt_agent.opt_dims.cvars.nary=['position|velocity'] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null
python scripts/main.py calvin_open_drawer_dbopt_cvar --data-dir results/icra_bopt_exp_cod/is_cod_p5_n01_10_05_10_wc_sR_3 --run-prefix icra_is_bopt_cod_p5_n01_10_05_10_wc_sR --deep-eval 50 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/calvin_open_drawer_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.0 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=rotation bopt_agent.opt_dims.cvars.nary=['position|velocity'] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null
python scripts/main.py calvin_open_drawer_dbopt_cvar --data-dir results/icra_bopt_exp_cod/is_cod_p5_n01_10_05_10_wc_sR_4 --run-prefix icra_is_bopt_cod_p5_n01_10_05_10_wc_sR --deep-eval 50 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/calvin_open_drawer_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.0 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=rotation bopt_agent.opt_dims.cvars.nary=['position|velocity'] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null
python scripts/main.py calvin_open_drawer_dbopt_cvar --data-dir results/icra_bopt_exp_cod/is_cod_p5_n01_10_05_10_wc_sR_5 --run-prefix icra_is_bopt_cod_p5_n01_10_05_10_wc_sR --deep-eval 50 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/calvin_open_drawer_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.0 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=rotation bopt_agent.opt_dims.cvars.nary=['position|velocity'] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null
python scripts/main.py calvin_open_drawer_dbopt_cvar --data-dir results/icra_bopt_exp_cod/is_cod_p5_n01_10_05_10_wc_sR_6 --run-prefix icra_is_bopt_cod_p5_n01_10_05_10_wc_sR --deep-eval 50 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/calvin_open_drawer_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.0 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=rotation bopt_agent.opt_dims.cvars.nary=['position|velocity'] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null

# Means and covar
python scripts/main.py calvin_open_drawer_dbopt_cvar --data-dir results/icra_bopt_exp_cod/is_cod_p5_n01_10_05_10_wpe_sR_1 --run-prefix icra_is_bopt_cod_p5_n01_10_05_10_wpe_sR --deep-eval 50 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/calvin_open_drawer_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=eigen bopt_agent.opt_dims.cvars.nary=['position|velocity'] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null
python scripts/main.py calvin_open_drawer_dbopt_cvar --data-dir results/icra_bopt_exp_cod/is_cod_p5_n01_10_05_10_wpe_sR_2 --run-prefix icra_is_bopt_cod_p5_n01_10_05_10_wpe_sR --deep-eval 50 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/calvin_open_drawer_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=eigen bopt_agent.opt_dims.cvars.nary=['position|velocity'] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null
python scripts/main.py calvin_open_drawer_dbopt_cvar --data-dir results/icra_bopt_exp_cod/is_cod_p5_n01_10_05_10_wpe_sR_3 --run-prefix icra_is_bopt_cod_p5_n01_10_05_10_wpe_sR --deep-eval 50 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/calvin_open_drawer_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=eigen bopt_agent.opt_dims.cvars.nary=['position|velocity'] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null
python scripts/main.py calvin_open_drawer_dbopt_cvar --data-dir results/icra_bopt_exp_cod/is_cod_p5_n01_10_05_10_wpe_sR_4 --run-prefix icra_is_bopt_cod_p5_n01_10_05_10_wpe_sR --deep-eval 50 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/calvin_open_drawer_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=eigen bopt_agent.opt_dims.cvars.nary=['position|velocity'] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null
python scripts/main.py calvin_open_drawer_dbopt_cvar --data-dir results/icra_bopt_exp_cod/is_cod_p5_n01_10_05_10_wpe_sR_5 --run-prefix icra_is_bopt_cod_p5_n01_10_05_10_wpe_sR --deep-eval 50 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/calvin_open_drawer_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=eigen bopt_agent.opt_dims.cvars.nary=['position|velocity'] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null
python scripts/main.py calvin_open_drawer_dbopt_cvar --data-dir results/icra_bopt_exp_cod/is_cod_p5_n01_10_05_10_wpe_sR_6 --run-prefix icra_is_bopt_cod_p5_n01_10_05_10_wpe_sR --deep-eval 50 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/calvin_open_drawer_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=eigen bopt_agent.opt_dims.cvars.nary=['position|velocity'] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null

python scripts/main.py calvin_open_drawer_dbopt_cvar --data-dir results/icra_bopt_exp_cod/is_cod_p5_n01_10_05_10_wpc_sR_1 --run-prefix icra_is_bopt_cod_p5_n01_10_05_10_wpc_sR --deep-eval 50 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/calvin_open_drawer_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=rotation bopt_agent.opt_dims.cvars.nary=['position|velocity'] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null
python scripts/main.py calvin_open_drawer_dbopt_cvar --data-dir results/icra_bopt_exp_cod/is_cod_p5_n01_10_05_10_wpc_sR_2 --run-prefix icra_is_bopt_cod_p5_n01_10_05_10_wpc_sR --deep-eval 50 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/calvin_open_drawer_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=rotation bopt_agent.opt_dims.cvars.nary=['position|velocity'] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null
python scripts/main.py calvin_open_drawer_dbopt_cvar --data-dir results/icra_bopt_exp_cod/is_cod_p5_n01_10_05_10_wpc_sR_3 --run-prefix icra_is_bopt_cod_p5_n01_10_05_10_wpc_sR --deep-eval 50 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/calvin_open_drawer_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=rotation bopt_agent.opt_dims.cvars.nary=['position|velocity'] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null
python scripts/main.py calvin_open_drawer_dbopt_cvar --data-dir results/icra_bopt_exp_cod/is_cod_p5_n01_10_05_10_wpc_sR_4 --run-prefix icra_is_bopt_cod_p5_n01_10_05_10_wpc_sR --deep-eval 50 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/calvin_open_drawer_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=rotation bopt_agent.opt_dims.cvars.nary=['position|velocity'] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null
python scripts/main.py calvin_open_drawer_dbopt_cvar --data-dir results/icra_bopt_exp_cod/is_cod_p5_n01_10_05_10_wpc_sR_5 --run-prefix icra_is_bopt_cod_p5_n01_10_05_10_wpc_sR --deep-eval 50 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/calvin_open_drawer_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=rotation bopt_agent.opt_dims.cvars.nary=['position|velocity'] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null
python scripts/main.py calvin_open_drawer_dbopt_cvar --data-dir results/icra_bopt_exp_cod/is_cod_p5_n01_10_05_10_wpc_sR_6 --run-prefix icra_is_bopt_cod_p5_n01_10_05_10_wpc_sR --deep-eval 50 --ckpt-freq 1 --wandb --overrides bopt_agent.gmm.model=models/gmm/calvin_open_drawer_5p.npy env.noise.position.variance=0.01 bopt_agent.acq_optimizer=hpo bopt_agent.prior_range=0.1 bopt_agent.mean_range=0.05 bopt_agent.sigma_range=0.1 bopt_agent.opt_dims.means=[position] bopt_agent.opt_dims.cvars.type=rotation bopt_agent.opt_dims.cvars.nary=['position|velocity'] bopt_agent.opt_dims.cvars.unary=null bopt_agent.num_training_cycles=500 bopt_agent.incumbent_eval=null
