import sys
import subprocess
import atexit
import time
from argparse import ArgumentParser
from functools import partial
import os
def run_new_command(id):

    new_command = ["python", "playground.py","--id",str(id)]  # 替换为你想要执行的命令和参数

    subprocess.Popen(new_command)

# atexit.register(partial(run_new_command, id=0))

#sleep 10

print("Hello, World!")
parser = ArgumentParser()
parser.add_argument("--id", type=int, default=0)
args = parser.parse_args()
print(args.id)
time.sleep(1)

run_new_command(args.id+1)
sys.exit()
