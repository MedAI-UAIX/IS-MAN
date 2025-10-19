from pathlib import Path

readme_content = """# IS-MAN

**IS-MAN: Intelligent Sonographic-MANipulation**  
*A First-in-Clinical Large-Scale Study of LLM-Driven Embodied Intelligence for Fully Autonomous Robotic Ultrasound*

---

## 🧠 Introduction

**IS-MAN** aims to establish a fully autonomous robotic ultrasound system powered by **large language models (LLMs)** and **multimodal embodied intelligence**.  
It integrates robot control, ultrasound image understanding, and clinical decision assistance, exploring a new paradigm for *LLM-driven embodied medicine*.

---

## 📁 Repository Structure

IS-MAN/
├── Azure_Kinect_ROS_Driver/ # Azure Kinect sensor integration
├── OrbbecSDK_ROS1/ # Orbbec depth camera SDK for ROS1
├── aruco_ros/ # ArUco marker-based localization and calibration
├── easy_handeye/ # Eye-in-hand calibration utilities
├── franka/ # Franka Emika robot configuration
├── franka_ros/ # ROS driver for Franka robot
├── franka_ros_interface/ # High-level ROS control interfaces
├── robotiq_ft_sensor/ # Robotiq force-torque sensor driver
├── vision_visp/ # Visual servoing modules (ViSP)
├── franka.rviz # RViz visualization configuration
└── README.md # Project documentation


---

## ⚙️ System Requirements

- **Ubuntu 20.04 / ROS Noetic**
- **Franka ROS driver v0.9+**
- **ViSP / OpenCV / PCL**
- **Azure Kinect SDK / Orbbec SDK**
- **Python ≥ 3.8**

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone git@github.com:MedAI-UAIX/IS-MAN.git
cd IS-MAN

### 2. Setup ROS Workspace
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
ln -s /path/to/IS-MAN .
cd ..
catkin_make
source devel/setup.bash

### 3. Launch Example Modules
roslaunch franka_ros_interface franka_control.launch
roslaunch easy_handeye calibrate.launch



