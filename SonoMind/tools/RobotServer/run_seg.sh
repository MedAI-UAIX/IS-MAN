#!/bin/bash

# 设置 GPU 可见性
export CUDA_VISIBLE_DEVICES=0  # 选择可用的 GPU 设备，可以根据需要更改
# conda activate  pytorch_gpu_3.8

cd /home/usai/auto_RUSS/R_02_Unet

# 执行Python脚本
# human
# /home/uax/.conda/envs/pytorch_gpu_3.8/bin/python "/home/uax/LiMD_example/Robot_arm/R_02_Unet/inceptionext_seg.py"

#human  canon和mindray微调
#  /home/uax/.conda/envs/pytorch_gpu_3.8/bin/python "/home/uax/LiMD_example/Robot_arm/R_02_Unet/convnext_seg-canon_mindray-topic.py"

/home/usai/miniconda3/bin/python3.8 "/home/usai/auto_RUSS/R_02_Unet/Swin_Unet phantom topic_lxxx.py"
# /home/usai/miniconda3/bin/python3.8 "/home/usai/auto_RUSS/R_02_Unet/Swin_Unet phantom topic_lxxx_thyroid_car.py"

#phantom
# /home/uax/.conda/envs/pytorch_gpu_3.8/bin/python "/home/uax/LiMD_example/Robot_arm/R_02_Unet/Swin_Unet.py"
# /home/uax/.conda/envs/pytorch_gpu_3.8/bin/python "/home/uax/LiMD_example/Robot_arm/R_02_Unet/Swin_Unet phantom topic.py"

