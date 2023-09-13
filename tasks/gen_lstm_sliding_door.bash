python scripts/gen_bc.py demos_sliding_door/demo_0* --modalities position --out models/bc/sliding_door_lstm_geom_100
python scripts/gen_bc.py demos_sliding_door/demo_0* --modalities position doorpos --out models/bc/sliding_door_lstm_js_100
python scripts/gen_bc.py demos_sliding_door/demo_0* --modalities position force --out models/bc/sliding_door_lstm_f_100
python scripts/gen_bc.py demos_sliding_door/demo_0* --modalities position torque --out models/bc/sliding_door_lstm_t_100
python scripts/gen_bc.py demos_sliding_door/demo_0* --modalities position force torque --out models/bc/sliding_door_lstm_ft_100
