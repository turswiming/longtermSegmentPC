import os
import subprocess
val = ["0"]

learning_rate = [
    (100.00,0.0,0.0,0.0),
    (10.00,0.0,0.0,0.0),
    (1.00,0.0,0.0,0.0),
    (0.10,0.0,0.0,0.0),
    (0.01,0.0,0.0,0.0),
    (0.001,0.0,0.0,0.0),
    (0.0001,0.0,0.0,0.0),
    ]
for val_data in val:
    for lr in learning_rate:
        
        ablationtype = f"opt3d_lr{lr[0]}"
        print('ablationtype',ablationtype)
        os.makedirs(f'../log/{ablationtype}', exist_ok=True)
        path = f'../log/{ablationtype}/{val_data}.txt'
        if os.path.exists(path):
            continue
        command_list = [
            "python main.py",
            f"GWM.FOCUS_DATA {val_data}",
            f"GWM.LOSS_MULT.OPT {lr[0]}",
            f"GWM.LOSS_MULT.OPT2 {lr[1]}",
            f"GWM.LOSS_MULT.TRAJ {lr[2]}",
            f"GWM.LOSS_MULT.TRAJ3 {lr[3]}",
            f"ABLATION.NAME {ablationtype}"
        ]
        command = " ".join(command_list)
        print(command)
        subprocess.run(command, shell=True)