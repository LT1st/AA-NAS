@echo off
setlocal enabledelayedexpansion

rem 循环执行 5 次
for /l %%i in (1, 1, 5) do (
    rem 复制 AA_data 文件夹并重命名为 AA_data_i
    xcopy /e /i AA_data AA_data_%%i

    rem 运行 run_AA.py
    python run_AA.py

    rem 运行 run_NAS.py
    python run_NAS.py

    rem 运行 train_NAS_result.py
    python train_NAS_result.py
)