python scripts/gen_bc.py demos_drawer/demo_00* --modalities position --out models/bc/drawer_lstm_geom
python scripts/gen_bc.py demos_drawer/demo_00* --modalities position drawerpos --out models/bc/drawer_lstm_js
python scripts/gen_bc.py demos_drawer/demo_00* --modalities position force --out models/bc/drawer_lstm_f
python scripts/gen_bc.py demos_drawer/demo_00* --modalities position torque --out models/bc/drawer_lstm_t
python scripts/gen_bc.py demos_drawer/demo_00* --modalities position force torque --out models/bc/drawer_lstm_ft
