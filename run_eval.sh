#!/bin/bash

# 设置日志文件
LOG_FILE="eval_$(date +%Y%m%d_%H%M%S).log"

# 激活conda环境（如果需要的话）
# source /path/to/your/conda/bin/activate
# conda activate your_env_name

# 使用nohup运行评估脚本，并将输出重定向到日志文件
nohup python eval.py > $LOG_FILE 2>&1 &

# 获取进程ID
PID=$!

# 将进程ID写入文件
echo $PID > eval.pid

echo "评估脚本已在后台启动，进程ID: $PID"
echo "日志文件: $LOG_FILE"
echo "可以使用 'tail -f $LOG_FILE' 查看实时日志"
echo "使用 'kill $(cat eval.pid)' 停止进程" 