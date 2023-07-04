#!/bin/bash

# 循环执行 5 次
for i in {1..5}
do
    # 复制 AA_data 文件夹并重命名为 AA_data_i
    cp -r AA_data AA_data_$i

    # 运行 run_AA.py
    python3 run_AA.py

    # 运行 run_NAS.py
    python3 run_NAS.py

    # 运行 train_NAS_result.py
    python3 train_NAS_result.py
done