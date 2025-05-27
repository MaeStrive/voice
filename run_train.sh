#!/bin/bash

# 设置日志文件
LOG_FILE="train_$(date +%Y%m%d_%H%M%S).log"

# 激活conda环境（如果需要的话）
# source /path/to/your/conda/bin/activate
# conda activate your_env_name

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0

# 使用nohup运行训练脚本，并将输出重定向到日志文件
nohup python train.py > $LOG_FILE 2>&1 &

# 获取进程ID
PID=$!

# 将进程ID写入文件
echo $PID > train.pid

echo "训练脚本已在后台启动，进程ID: $PID"
echo "日志文件: $LOG_FILE"
echo "可以使用 'tail -f $LOG_FILE' 查看实时日志"
echo "使用 'kill $(cat train.pid)' 停止进程" 