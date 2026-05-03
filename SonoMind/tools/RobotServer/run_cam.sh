#!/bin/bash
# 设置 GPU 可见性
export CUDA_VISIBLE_DEVICES=0  # 选择可用的 GPU 设备，可以根据需要更改
# source activate  pytorch_gpu_3.8

cd /home/usai/auto_RUSS/R_03_keypoint

# 执行Python脚本
# /home/usai/miniconda3/envs/pytorch_gpu_3.8/bin/python /home/usai/auto_RUSS/R_03_keypoint/00_detection_v4_cvbridge_点云_奥比中光.py    # 手动点击
# /home/usai/miniconda3/envs/pytorch_gpu_3.8/bin/python /home/usai/auto_RUSS/R_03_keypoint/00_detection_v4_cvbridge_点云_奥比中光_yolov8.py    # 自动检测
/home/usai/miniconda3/bin/python3.8 /home/usai/auto_RUSS/R_03_keypoint/00_detection_v4_cvbridge_点云_奥比中光_yolov8.py