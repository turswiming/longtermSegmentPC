import os
import subprocess
val = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15",]
learning_rate = [
    (1000.0,0.0,0.0,0.0,"opt3d_only"),
    (0.0,0.0,0.00005,0.0,"traj3d_only"),
    (1000.00,0.0,0.00005,0.0,"opt3d_traj3d"),
    (0.0,0.03,0.0,0.00005,"opt2d_traj2d"),
    ]
for i,val_data in enumerate(val):
    for lr in learning_rate:
        
        ablationtype = lr[-1]
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
            f"ABLATION.NAME {ablationtype}",
            f"ABLATION.RESULTSAVEPATH {path}",
        ]
        command = " ".join(command_list)
        print(command)
        subprocess.run(command, shell=True)