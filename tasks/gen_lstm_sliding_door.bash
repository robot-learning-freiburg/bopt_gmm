python scripts/gen_bc.py demos_sliding_door/demo_0* --modalities position --out models/bc/sliding_door_lstm_geom
python scripts/gen_bc.py demos_sliding_door/demo_0* --modalities position doorpos --out models/bc/sliding_door_lstm_js
python scripts/gen_bc.py demos_sliding_door/demo_0* --modalities position force --out models/bc/sliding_door_lstm_f
python scripts/gen_bc.py demos_sliding_door/demo_0* --modalities position torque --out models/bc/sliding_door_lstm_t
python scripts/gen_bc.py demos_sliding_door/demo_0* --modalities position force torque --out models/bc/sliding_door_lstm_ft
