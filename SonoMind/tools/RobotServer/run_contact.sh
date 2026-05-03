#!/bin/bash
# 设置 GPU 可见性
export CUDA_VISIBLE_DEVICES=0  # 选择可用的 GPU 设备，可以根据需要更改
# conda activate  pytorch_gpu_3.8

# 执行Python脚本
# /home/uax/.conda/envs/pytorch_gpu_3.8/bin/python "/home/uax/LiMD_example/Robot_arm/AutoScanning/contact_detecting-topic-two.py"
# /home/uax/.conda/envs/pytorch_gpu_3.8/bin/python "/home/uax/LiMD_example/Robot_arm/AutoScanning/contact_detecting-topic-two_lxx.py"    # 5份
# /home/usai/miniconda3/envs/pytorch_gpu_3.8/bin/python "/home/usai/auto_RUSS/AutoScanning/contact_detecting-topic-two_v2.py"    # 24份
/home/usai/miniconda3/bin/python3.8 /home/usai/auto_RUSS/AutoScanning/contact_detecting-topic-two_v2.py