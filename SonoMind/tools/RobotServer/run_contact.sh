#!/bin/bash
# run_seg.sh
set -e  # Exit on error

# Get the directory where the script is located (RobotServer directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Set GPU visibility
export CUDA_VISIBLE_DEVICES=0

# Switch to the corresponding module directory (using the relative path)
cd "$PROJECT_ROOT/AutoScanning"

# Use Python from environment variables (or activate the environment via conda activate)
# It is recommended to activate the conda environment first

# Execute the Python script (using the relative path)
python "contact_detecting-topic-two_v2.py"