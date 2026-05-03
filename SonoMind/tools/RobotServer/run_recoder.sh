#!/bin/bash
# 设置 GPU 可见性
export CUDA_VISIBLE_DEVICES=0  # 选择可用的 GPU 设备，可以根据需要更改
# conda activate  pytorch_gpu_3.8

cd /home/usai/auto_RUSS/R_20_Recording_Status

# 执行Python脚本
# /home/usai/miniconda3/envs/pytorch_gpu_3.8/bin/python /home/usai/auto_RUSS/R_20_Recording_Status/recording.py
/home/usai/miniconda3/bin/python3.8 /home/usai/auto_RUSS/R_20_Recording_Status/recording.py
