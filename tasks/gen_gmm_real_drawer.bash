python scripts/gen_gmm.py demos_real_drawer/demo_*.npz --generator em --max-iter 500 --tol-cutting 0.05 --n-priors 7 --n-init 20 --modalities position --normalize --out models/gmm/real_drawer_7p.npy
python scripts/gen_gmm.py demos_real_drawer/demo_*.npz --generator em --max-iter 500 --tol-cutting 0.05 --n-priors 5 --n-init 20 --modalities position --normalize --out models/gmm/real_drawer_5p.npy
python scripts/gen_gmm.py demos_real_drawer/demo_*.npz --generator em --max-iter 500 --tol-cutting 0.05 --n-priors 3 --n-init 20 --modalities position --normalize --out models/gmm/real_drawer_3p.npy
