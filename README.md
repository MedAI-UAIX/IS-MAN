# IS-MAN

**IS-MAN: Intelligent Sonographic-MANipulation**  
*A First-in-Clinical Large-Scale Study of LLM-Driven Embodied Intelligence for Fully Autonomous Robotic Ultrasound*

---

## 🧠 Introduction

**IS-MAN** aims to establish a fully autonomous robotic ultrasound system powered by **large language models (LLMs)** and **multimodal embodied intelligence**.  
It integrates robot control, ultrasound image understanding, and clinical decision assistance, exploring a new paradigm for *LLM-driven embodied medicine*.

---

## 📁 Repository Structure

```text
IS-MAN/
├── Azure_Kinect_ROS_Driver/      # Azure Kinect sensor integration
├── OrbbecSDK_ROS1/               # Orbbec depth camera SDK for ROS1
├── aruco_ros/                    # ArUco marker-based localization and calibration
├── easy_handeye/                 # Eye-in-hand calibration utilities
├── franka/                       # Franka Emika robot configuration
├── franka_ros/                   # ROS driver for Franka robot
├── franka_ros_interface/         # High-level ROS control interfaces
├── robotiq_ft_sensor/            # Robotiq force-torque sensor driver
├── vision_visp/                  # Visual servoing modules (ViSP)
├── franka.rviz                   # RViz visualization configuration
└── README.md                     # Project documentation
```

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
```

### 2. Setup ROS Workspace
```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
ln -s /path/to/IS-MAN .
cd ..
catkin_make
source devel/setup.bash
```

### 3. Launch Example Modules
```bash
roslaunch franka_ros_interface franka_control.launch
roslaunch easy_handeye calibrate.launch
```

> Adjust parameters in each package’s `config/` folder for device calibration, camera parameters, and robot IPs.

---

## 🧩 Research Overview

**IS-MAN (Intelligent Sonographic-MANipulation)** focuses on building an embodied AI system for autonomous ultrasound, integrating:

- Autonomous robotic ultrasound probe path planning  
- Adaptive contact and force control  
- Real-time ultrasound image understanding  
- Cross-modal alignment (visual + acoustic)  
- LLM-based clinical reasoning and strategy scheduling  

This project is part of the **MedAI-UAIX Laboratory** initiative in embodied AI for clinical ultrasound and robotic medicine.

---

## 📄 License

This project is released under the **Apache 2.0 License**.  
The full source code will be **open-sourced after publication**.

---

## 👥 Authors & Acknowledgements

**MedAI-UAIX Laboratory**  
Principal contributors: *Franka-Med et al.*

We gratefully acknowledge all collaborators in robotics, ultrasonography, and clinical AI research.

---

## 📚 Citation

If you find this work useful, please cite:

```
Franka-Med et al.  
IS-MAN: A First-in-Clinical Large-Scale Study of LLM-Driven Embodied Intelligence for Fully Autonomous Robotic Ultrasound.  
2025 (under review)
```
