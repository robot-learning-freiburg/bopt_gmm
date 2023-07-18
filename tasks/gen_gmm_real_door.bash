python scripts/gen_gmm.py demos_real_door/demo_*.npz --generator em --max-iter 500 --tol-cutting 0.05 --n-priors 7 --n-init 20 --modalities position --normalize --out models/gmm/real_door_7p.npy
python scripts/gen_gmm.py demos_real_door/demo_*.npz --generator em --max-iter 500 --tol-cutting 0.05 --n-priors 5 --n-init 20 --modalities position --normalize --out models/gmm/real_door_5p.npy
python scripts/gen_gmm.py demos_real_door/demo_*.npz --generator em --max-iter 500 --tol-cutting 0.05 --n-priors 3 --n-init 20 --modalities position --normalize --out models/gmm/real_door_3p.npy
