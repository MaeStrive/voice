@echo off
setlocal

:: 设置日志文件名
set LOG_FILE=eval_%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log
set LOG_FILE=%LOG_FILE: =0%

:: 启动Python脚本
start /B python eval.py > %LOG_FILE% 2>&1

:: 获取进程ID
for /f "tokens=2" %%a in ('tasklist ^| findstr "python.exe"') do set PID=%%a

:: 将进程ID写入文件
echo %PID% > eval.pid

echo 评估脚本已在后台启动，进程ID: %PID%
echo 日志文件: %LOG_FILE%
echo 可以使用 'type %LOG_FILE%' 查看日志
echo 使用 'taskkill /F /PID %PID%' 停止进程

endlocal 