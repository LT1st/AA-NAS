# -*- coding: utf-8 -*-

import subprocess
import time

iter_times = 5      # 总迭代次数
waiting_time = 10    # 等待时间

baseFile_dataset_NAS = 'data'       # NAS数据集
baseFile_dataset_AA = 'AA_data'     # AA的数据

for i in range(1, 1+iter_times):
    this_dataset_NAS = baseFile_dataset_NAS
    this_dataset_AA  = baseFile_dataset_AA + '_' + str(i)

    # 复制 AA_data 文件夹并重命名为 AA_data_i
    while True:
        # 区分不同次的AA数据集
        result = subprocess.run(['cp', '-r', 'AA_data', this_dataset_AA],
                                capture_output=True)
        # 检查子进程的返回值
        if result.returncode == 0:
            print('子进程执行成功', i)
            break  # 子进程执行成功，跳出while循环
        else:
            pass
            print('子进程执行失败，重试中...')
        # 等待一段时间后重试
        time.sleep(waiting_time)

    # 运行 run_AA.py
    while True:
        # 执行Python文件
        result = subprocess.run(['python3', 'run_AA.py', this_dataset_AA],
                                capture_output=True)
        # 检查子进程的返回值
        if result.returncode == 0:
            print('子进程执行成功', i)
            break  # 子进程执行成功，跳出while循环
        else:
            pass
            # print('子进程run_AA执行失败，重试中...')
        # 等待一段时间后重试
        time.sleep(waiting_time)

    # 运行 run_NAS.py
    subprocess.run(['python3', 'run_NAS.py'])

    # 运行 train_NAS_result.py
    subprocess.run(['python3', 'train_NAS_result.py'])