@echo off
REM 一键训练_Krea2风格LoRA.bat — RTX 3060 12G / 32G 内存专用
REM 用前先改下面三个路径！训练前关掉所有占显存/内存的程序（浏览器、ComfyUI 等）

set AI_TOOLKIT_DIR=C:\lora-training\ai-toolkit
set CONFIG_FILE=%AI_TOOLKIT_DIR%\config\krea2_style_lora_3060.yml
set LOG_FILE=%AI_TOOLKIT_DIR%\..\train_run.log

REM ===== 显存预检：可用显存必须 >= 9GB =====
for /f "tokens=1,2 delims=, " %%a in ('nvidia-smi --query-gpu=memory.free --format=csv,noheader ^| findstr /r "^[0-9]"') do set FREE_VRAM=%%a
echo 当前可用显存: %FREE_VRAM% MiB（需要 >= 9000）
if %FREE_VRAM% LSS 9000 (
    echo.
    echo [阻断] 显存不足！请关闭浏览器 / ComfyUI / 游戏等后重试。
    pause
    exit /b 1
)

REM ===== 环境保险 =====
REM PYTHONPATH 清空：防外部包路径污染 venv
set PYTHONPATH=
REM 必带！跳过锁页（Windows 内核非分页池会被 16G 权重打爆，之后一切 CUDA 分配报 OOM）
set AI_TOOLKIT_NO_PIN=1

echo.
echo 开始训练，日志: %LOG_FILE%
echo 训练中不要开大型程序（32G 内存已贴满，开了会触发磁盘分页拖慢训练）
echo.
cd /d "%AI_TOOLKIT_DIR%"
.venv\Scripts\python.exe run.py %CONFIG_FILE% > "%LOG_FILE%" 2>&1

echo.
echo 训练进程退出（退出码 %ERRORLEVEL%）。
echo 日志末尾 30 行:
powershell -NoProfile -Command "Get-Content '%LOG_FILE%' -Tail 30"
pause
