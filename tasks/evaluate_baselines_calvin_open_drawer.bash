python scripts/job_runner.py 15 "python scripts/main.py calvin_open_drawer_scenario --mode eval-gmm --deep-eval 100 --overrides env.noise.position.variance=0.00 bopt_agent.gmm.model=models/gmm/calvin_open_drawer_7p.npy --eval-out /tmp/baselines_calvin_open_drawer_7p_n00.csv" \
                                "python scripts/main.py calvin_open_drawer_scenario --mode eval-gmm --deep-eval 100 --overrides env.noise.position.variance=0.00 bopt_agent.gmm.model=models/gmm/calvin_open_drawer_5p.npy --eval-out /tmp/baselines_calvin_open_drawer_5p_n00.csv" \
                                "python scripts/main.py calvin_open_drawer_scenario --mode eval-gmm --deep-eval 100 --overrides env.noise.position.variance=0.00 bopt_agent.gmm.model=models/gmm/calvin_open_drawer_3p.npy --eval-out /tmp/baselines_calvin_open_drawer_3p_n00.csv" \
                                "python scripts/main.py calvin_open_drawer_scenario --mode eval-gmm --deep-eval 100 --overrides env.noise.position.variance=0.01 bopt_agent.gmm.model=models/gmm/calvin_open_drawer_7p.npy --eval-out /tmp/baselines_calvin_open_drawer_7p_n01.csv" \
                                "python scripts/main.py calvin_open_drawer_scenario --mode eval-gmm --deep-eval 100 --overrides env.noise.position.variance=0.01 bopt_agent.gmm.model=models/gmm/calvin_open_drawer_5p.npy --eval-out /tmp/baselines_calvin_open_drawer_5p_n01.csv" \
                                "python scripts/main.py calvin_open_drawer_scenario --mode eval-gmm --deep-eval 100 --overrides env.noise.position.variance=0.01 bopt_agent.gmm.model=models/gmm/calvin_open_drawer_3p.npy --eval-out /tmp/baselines_calvin_open_drawer_3p_n01.csv" \
                                "python scripts/main.py calvin_open_drawer_scenario --mode eval-gmm --deep-eval 100 --overrides env.noise.position.variance=0.02 bopt_agent.gmm.model=models/gmm/calvin_open_drawer_7p.npy --eval-out /tmp/baselines_calvin_open_drawer_7p_n02.csv" \
                                "python scripts/main.py calvin_open_drawer_scenario --mode eval-gmm --deep-eval 100 --overrides env.noise.position.variance=0.02 bopt_agent.gmm.model=models/gmm/calvin_open_drawer_5p.npy --eval-out /tmp/baselines_calvin_open_drawer_5p_n02.csv" \
                                "python scripts/main.py calvin_open_drawer_scenario --mode eval-gmm --deep-eval 100 --overrides env.noise.position.variance=0.02 bopt_agent.gmm.model=models/gmm/calvin_open_drawer_3p.npy --eval-out /tmp/baselines_calvin_open_drawer_3p_n02.csv"
python scripts/csv_fusion.py data/baselines_calvin_open_drawer.csv /tmp/baselines_calvin_open_drawer_*.csv
