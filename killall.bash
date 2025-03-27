#!/bin/bash

# 获取所有包含 "python main.py" 和 ".pth" 的进程 ID
pids=$(ps aux | grep 'python main.py' | awk '{print $2}')

# 检查是否有匹配的进程
if [ -z "$pids" ]; then
  echo "No matching processes found."
else
  # 杀死所有匹配的进程
  echo "Killing the following processes: $pids"
  kill -9 $pids
fi