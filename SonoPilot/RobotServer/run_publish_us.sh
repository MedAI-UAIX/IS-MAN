#!/bin/bash
export CUDA_VISIBLE_DEVICES=0  # 选择可用的 GPU 设备，可以根据需要更改

/home/usai/miniconda3/bin/python3.8 /home/usai/IS-MAN/SonoPilot/RobotServer/publish_US_cv2.py