@echo off
setlocal

:: 设置日志文件名
set LOG_FILE=train_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log
set LOG_FILE=%LOG_FILE: =0%

:: 设置CUDA设备
set CUDA_VISIBLE_DEVICES=0

:: 设置环境变量
set PYTHONPATH=%PYTHONPATH%;%CD%

:: 启动Python训练脚本
python train.py > %LOG_FILE% 2>&1

echo 训练脚本已启动，日志文件: %LOG_FILE%
echo 可以使用 'type %LOG_FILE%' 查看日志

endlocal 