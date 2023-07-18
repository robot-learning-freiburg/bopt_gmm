python scripts/gen_gmm.py demos_peg/demo_*.npz --generator em --max-iter 500 --tol-cutting 0.05 --n-priors 3 --n-init 20 --modalities position --out models/gmm/peg_3p.npy
python scripts/gen_gmm.py demos_peg/demo_*.npz --generator em --max-iter 500 --tol-cutting 0.05 --n-priors 5 --n-init 20 --modalities position --out models/gmm/peg_5p.npy
python scripts/gen_gmm.py demos_peg/demo_*.npz --generator em --max-iter 500 --tol-cutting 0.05 --n-priors 7 --n-init 20 --modalities position --out models/gmm/peg_7p.npy
python scripts/gen_gmm.py demos_peg/demo_*.npz --generator em --max-iter 500 --tol-cutting 0.05 --n-priors 3 --n-init 20 --modalities position force --normalize --out models/gmm/peg_3p_f.npy
python scripts/gen_gmm.py demos_peg/demo_*.npz --generator em --max-iter 500 --tol-cutting 0.05 --n-priors 5 --n-init 20 --modalities position force --normalize --out models/gmm/peg_5p_f.npy
python scripts/gen_gmm.py demos_peg/demo_*.npz --generator em --max-iter 500 --tol-cutting 0.05 --n-priors 7 --n-init 20 --modalities position force --normalize --out models/gmm/peg_7p_f.npy
python scripts/gen_gmm.py demos_peg/demo_*.npz --generator em --max-iter 500 --tol-cutting 0.05 --n-priors 3 --n-init 20 --modalities position torque --normalize --out models/gmm/peg_3p_t.npy
python scripts/gen_gmm.py demos_peg/demo_*.npz --generator em --max-iter 500 --tol-cutting 0.05 --n-priors 5 --n-init 20 --modalities position torque --normalize --out models/gmm/peg_5p_t.npy
python scripts/gen_gmm.py demos_peg/demo_*.npz --generator em --max-iter 500 --tol-cutting 0.05 --n-priors 7 --n-init 20 --modalities position torque --normalize --out models/gmm/peg_7p_t.npy
