import os
import subprocess
val = ["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19"]

learning_rate = [(0.0,0.03,0.0,0.0),(0.03,0.0,0.0,0.0),(0.03,0.0,0.00005,0.0),(0.03,0.0,0.0,0.00005),(0.0,0.0,0.00005,0.0),(0.0,0.0,0.0,0.00005)]
learning_rate = [(0.0,0.03,0.0005,0.0),(60.00,0.0,0.0005,0.0),]
for val_data in val:
    for lr in learning_rate:
        print(val_data, lr)
        use_opt = False
        use_opt2 = False
        use_traj = False
        use_traj3 = False
        if lr[0] > 0.0:
            use_opt =True
        if lr[1] > 0.0:
            use_opt2 = True
        if lr[2] > 0.0:
            use_traj = True
        if lr[3] > 0.0:
            use_traj3 = True
        if use_opt and use_traj:
            ablationtype = "opt3d_traj"
        elif use_opt2 and use_traj:
            ablationtype = "opt2d_traj"
        elif use_opt and use_traj3:
            ablationtype = "opt_traj_formula3"
        elif use_opt:
            ablationtype = "opt"
        elif use_opt2:
            ablationtype = "opt_formula2"
        elif use_traj:
            ablationtype = "traj_formula4"
        elif use_traj3:
            ablationtype = "traj_formula3"
        print('ablationtype',ablationtype)
        os.makedirs(f'../log/{ablationtype}', exist_ok=True)
        path = f'../log/{ablationtype}/{val_data}.txt'
        if os.path.exists(path):
            continue
        print(f"python main.py GWM.FOCUS_DATA {val_data} GWM.LOSS_MULT.OPT {lr[0]} GWM.LOSS_MULT.OPT2 {lr[1]} GWM.LOSS_MULT.TRAJ {lr[2]} GWM.LOSS_MULT.TRAJ3 {lr[3]}")
        subprocess.run(f"python main.py GWM.FOCUS_DATA {val_data} GWM.LOSS_MULT.OPT {lr[0]} GWM.LOSS_MULT.OPT2 {lr[1]} GWM.LOSS_MULT.TRAJ {lr[2]} GWM.LOSS_MULT.TRAJ3 {lr[3]}", shell=True)