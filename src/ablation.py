import os
import subprocess
val = ['blackswan',"car-shadow", 'bmx-trees', 'breakdance', 'camel', 'car-roundabout', 'cows', 'dance-twirl',
        'dog', 'drift-chicane', 'drift-straight', 'goat', 'horsejump-high', 'kite-surf', 'libby', 'motocross-jump',
        'paragliding-launch', 'parkour', 'scooter-black', 'soapbox']

learning_rate = [(0.0,0.03,0.0,0.0),(0.03,0.0,0.0,0.0),(0.03,0.0,0.00005,0.0),(0.03,0.0,0.0,0.00005),(0.0,0.0,0.00005,0.0),(0.0,0.0,0.0,0.00005)]

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
            ablationtype = "opt_traj_formula4"
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