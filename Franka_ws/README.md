# 🤖 Franka ROS Workspace for Robotic Control

## 📍 Overview

This directory contains the ROS workspace designed for developing and running control applications for the [Franka Emika Panda​](https://franka.de/) robot within the IS-MAN system.
It integrates the official Franka ROS packages (franka_ros) along with custom controllers and configuration files, providing a clean and robust environment for real-time robot control, motion execution, and controller development essential for autonomous ultrasound scanning.
This workspace serves as the foundational layer for [SonoPilot](https://github.com/MedAI-UAIX/IS-MAN/tree/main/SonoPilot), enabling seamless switching between Cartesian position control, impedance control, and hybrid force-position control.

### **Key Features**
- **Integrated Franka ROS Stack**  
  Includes franka_ros, moveit_franka, and related dependencies for streamlined setup.

- **Real-Time Controller Management**  
  Supports dynamic loading and switching of multiple controllers required for compliant ultrasound scanning.

-  **Perception-Action Loop**  
  Receives anatomical keypoints and lesion targets from upstream AI modules to dynamically adjust scanning trajectories.

-  **Native Safety Constraints**  
  Integrates libfranka safety controllers for real-time collision detection, joint limits, and Cartesian force/torque monitoring.

-  **Modular ROS Interfaces**  
  Provides standardized ROS topics/services for easy integration with autonomous state machines and path planners.



---

## 🛠️ Installation & Environment Setup

Follow these steps to set up the Franka ROS workspace.

### 📌 Dependencies
- [Ubuntu 20.04](https://releases.ubuntu.com/20.04/)
- [ROS Noetic](https://wiki.ros.org/noetic)
- [MoveIt](https://moveit.ai/)
- [Franka ROS Interface](https://github.com/frankarobotics/franka_ros)
- Python ≥3.7
- [libfranka ≥0.13.0](https://github.com/frankarobotics/libfranka)
- [OpenCV](https://opencv.org/)
- [FishBot ROS Installation Tool](https://github.com/fishros/install) (recommended for simplified ROS setup)

---

1. **Clone the repository**

```bash
git clone https://github.com/MedAI-UAIX/IS-MAN.git
cd IS-MAN/Franka_ws
```

2. **Create src directory**

```bash
mkdir -p ~/IS-MAN-main/Franka_ws/src
```

3. **Initialize the workspace**

```bash
cd ~/IS-MAN-main/Franka_ws/src
catkin_init_workspace
```

4. **Build the workspace**

```bash
cd ..
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
```

5. **Activate environment variables**

```bash
source IS-MAN-main/Franka_ws/devel/setup.bash
```

---

# 📬 Contact
For issues or improvements, please open an Issue.
