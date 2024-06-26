# python scripts/gen_gmm.py demos_sliding_door/demo_*.npz --generator em --max-iter 500 --tol-cutting 0.05 --n-priors 3 --n-init 20 --modalities position --out models/gmm/sd_3p.npy
# python scripts/gen_gmm.py demos_sliding_door/demo_*.npz --generator em --max-iter 500 --tol-cutting 0.05 --n-priors 5 --n-init 20 --modalities position --out models/gmm/sd_5p.npy
# python scripts/gen_gmm.py demos_sliding_door/demo_*.npz --generator em --max-iter 500 --tol-cutting 0.05 --n-priors 7 --n-init 20 --modalities position --out models/gmm/sd_7p.npy
# python scripts/gen_gmm.py demos_sliding_door/demo_*.npz --generator em --max-iter 500 --tol-cutting 0.05 --n-priors 3 --n-init 20 --modalities position force --normalize --out models/gmm/sd_3p_f.npy
# python scripts/gen_gmm.py demos_sliding_door/demo_*.npz --generator em --max-iter 500 --tol-cutting 0.05 --n-priors 5 --n-init 20 --modalities position force --normalize --out models/gmm/sd_5p_f.npy
# python scripts/gen_gmm.py demos_sliding_door/demo_*.npz --generator em --max-iter 500 --tol-cutting 0.05 --n-priors 7 --n-init 20 --modalities position force --normalize --out models/gmm/sd_7p_f.npy
# python scripts/gen_gmm.py demos_sliding_door/demo_*.npz --generator em --max-iter 500 --tol-cutting 0.05 --n-priors 3 --n-init 20 --modalities position torque --normalize --out models/gmm/sd_3p_t.npy
# python scripts/gen_gmm.py demos_sliding_door/demo_*.npz --generator em --max-iter 500 --tol-cutting 0.05 --n-priors 5 --n-init 20 --modalities position torque --normalize --out models/gmm/sd_5p_t.npy
# python scripts/gen_gmm.py demos_sliding_door/demo_*.npz --generator em --max-iter 500 --tol-cutting 0.05 --n-priors 7 --n-init 20 --modalities position torque --normalize --out models/gmm/sd_7p_t.npy
# python scripts/gen_gmm.py demos_sliding_door/demo_*.npz --generator em --max-iter 500 --tol-cutting 0.05 --n-priors 3 --n-init 20 --modalities position doorpos --out models/gmm/sd_3p_js.npy
# python scripts/gen_gmm.py demos_sliding_door/demo_*.npz --generator em --max-iter 500 --tol-cutting 0.05 --n-priors 5 --n-init 20 --modalities position doorpos --out models/gmm/sd_5p_js.npy
# python scripts/gen_gmm.py demos_sliding_door/demo_*.npz --generator em --max-iter 500 --tol-cutting 0.05 --n-priors 7 --n-init 20 --modalities position doorpos --out models/gmm/sd_7p_js.npy
python scripts/gen_gmm.py demos_sliding_door/demo_*.npz --generator seds --max-iter 500 --tol-cutting 0.05 --n-priors 3 --n-init 20 --modalities position --out models/gmm/sd_seds_3p.npy
python scripts/gen_gmm.py demos_sliding_door/demo_*.npz --generator seds --max-iter 500 --tol-cutting 0.05 --n-priors 5 --n-init 20 --modalities position --out models/gmm/sd_seds_5p.npy
python scripts/gen_gmm.py demos_sliding_door/demo_*.npz --generator seds --max-iter 500 --tol-cutting 0.05 --n-priors 7 --n-init 20 --modalities position --out models/gmm/sd_seds_7p.npy
python scripts/gen_gmm.py demos_sliding_door/demo_*.npz --generator seds --max-iter 500 --tol-cutting 0.05 --n-priors 3 --n-init 20 --modalities position force --normalize --out models/gmm/sd_seds_3p_f.npy
python scripts/gen_gmm.py demos_sliding_door/demo_*.npz --generator seds --max-iter 500 --tol-cutting 0.05 --n-priors 5 --n-init 20 --modalities position force --normalize --out models/gmm/sd_seds_5p_f.npy
python scripts/gen_gmm.py demos_sliding_door/demo_*.npz --generator seds --max-iter 500 --tol-cutting 0.05 --n-priors 7 --n-init 20 --modalities position force --normalize --out models/gmm/sd_seds_7p_f.npy
python scripts/gen_gmm.py demos_sliding_door/demo_*.npz --generator seds --max-iter 500 --tol-cutting 0.05 --n-priors 3 --n-init 20 --modalities position torque --normalize --out models/gmm/sd_seds_3p_t.npy
python scripts/gen_gmm.py demos_sliding_door/demo_*.npz --generator seds --max-iter 500 --tol-cutting 0.05 --n-priors 5 --n-init 20 --modalities position torque --normalize --out models/gmm/sd_seds_5p_t.npy
python scripts/gen_gmm.py demos_sliding_door/demo_*.npz --generator seds --max-iter 500 --tol-cutting 0.05 --n-priors 7 --n-init 20 --modalities position torque --normalize --out models/gmm/sd_seds_7p_t.npy
python scripts/gen_gmm.py demos_sliding_door/demo_*.npz --generator seds --max-iter 500 --tol-cutting 0.05 --n-priors 3 --n-init 20 --modalities position doorpos --out models/gmm/sd_seds_3p_js.npy
python scripts/gen_gmm.py demos_sliding_door/demo_*.npz --generator seds --max-iter 500 --tol-cutting 0.05 --n-priors 5 --n-init 20 --modalities position doorpos --out models/gmm/sd_seds_5p_js.npy
python scripts/gen_gmm.py demos_sliding_door/demo_*.npz --generator seds --max-iter 500 --tol-cutting 0.05 --n-priors 7 --n-init 20 --modalities position doorpos --out models/gmm/sd_seds_7p_js.npy
