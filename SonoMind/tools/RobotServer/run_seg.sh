#!/bin/bash
# run_seg.sh
set -e  # Exit on error

# Get the directory where the script is located (RobotServer directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Set GPU visibility
export CUDA_VISIBLE_DEVICES=0

# Switch to the corresponding module directory (using the relative path)
cd "$PROJECT_ROOT/R_02_Unet"

# Use Python from environment variables (or activate the environment via conda activate)
# It is recommended to activate the conda environment first

# 执行Python脚本（使用相对路径）
python "Swin_Unet phantom topic_lxxx.py"