# 🤖 Franka ROS Workspace for Robotic Control

## 📍 Overview

This directory contains the ROS workspace designed for developing and running control applications for the [Franka Emika Panda​](https://franka.de/) robot within the IS-MAN system.
It integrates the official Franka ROS packages (franka_ros) along with custom controllers and configuration files, providing a clean and robust environment for real-time robot control, motion execution, and controller development essential for autonomous ultrasound scanning.
This workspace serves as the foundational layer for SonoPilot, enabling seamless switching between Cartesian position control, impedance control, and hybrid force-position control.

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
- ROS Noetic
- MoveIt
- Franka ROS Interface
- Python ≥3.7
- libfranka ≥0.13.0
- [OpenCV](https://opencv.org/)
- [FishBot ROS Installation Tool](https://github.com/fishros/install) (recommended for simplified ROS setup)

---

1. **Create Conda Environment**

```bash
conda create -n franka python=3.10
conda activate franka
pip install -r requirements.txt
```



---

### 🚀 Controller Usage Examples

#### Cartesian Position Controller

```python
franka = MoveItFranka()
franka.switch_controllers(
    [franka.hybrid_force_position_controller_name],
    [franka.cartesian_position_controller_name]
)
```

#### Impedance Controller

```python
franka = MoveItFranka()
franka.hybrid_2_impedance()
franka.update_param(is_scaning=False, translational_stiffness=300, rotational_stiffness=50)
```

#### Hybrid Force-Position Controller

```python
franka = MoveItFranka()
franka.impedance_2_hybrid()
franka.update_param(is_scaning=True, ext_force=2)
```

#### Dynamic Controller Switching

```python
franka.switch_controllers(
    start_controllers=['desired_controller'],
    stop_controllers=['current_controller'],
    strictness=2,
    start_asap=True,
    timeout=1
)
```


---

### 📖 Controller Parameters Explained

#### `update_param()` in Impedance or Hybrid Control

```python
franka.update_param(
    is_scaning=True,
    translational_stiffness=300,
    rotational_stiffness=50,
    ext_force=2
)
```

| Parameter | Type | Description |
|----------|------|-------------|
| `is_scaning` | `bool` | Whether the system is in scanning mode. This flag helps control behavior such as contact force regulation and trajectory constraints. |
| `translational_stiffness` | `float` | Linear (XYZ) stiffness of the impedance controller, in N/m. Larger values make the robot resist external displacement more strongly. |
| `rotational_stiffness` | `float` | Angular (roll-pitch-yaw) stiffness in Nm/rad. Higher value = less compliant to rotation. |
| `ext_force` | `float` | Target contact force (in Newtons) for hybrid force-position control. Typically set between 1–5 N for soft tissue scanning. |

> 💡 Only the parameters relevant to the current control mode will be used.

---

#### `switch_controllers()`

```python
franka.switch_controllers(
    start_controllers=['hybrid_controller'],
    stop_controllers=['impedance_controller'],
    strictness=2,
    start_asap=True,
    timeout=1
)
```

| Parameter | Type | Description |
|----------|------|-------------|
| `start_controllers` | `List[str]` | List of controller names to start. |
| `stop_controllers` | `List[str]` | List of controller names to stop. |
| `strictness` | `int` | Controller switch strictness. `0`: best-effort, `1`: medium strict, `2`: fully strict. |
| `start_asap` | `bool` | Whether to start the new controller as soon as possible. |
| `timeout` | `float` | Timeout (in seconds) to complete the switch. Raises error if timeout exceeded. |

---
---


## License

All source code in this workspace is made available under the Apache-2.0 License.

---

## 🔗 Dependencies and Acknowledgements

We gratefully acknowledge the following open-source projects and frameworks:

- **libfranka** - Low-level robotic control library  
  https://github.com/frankaemika/libfranka

- **franka_ros** - Official ROS packages for Franka robots
  https://github.com/frankarobotics/franka_ros

- **MoveIt** - Motion planning framework
  https://moveit.ai/

- **ROS Noetic** - Robot Operating System
  https://wiki.ros.org/noetic

- **FishBot ROS Installation Tool** - Simplified ROS environment setup and dependency management
  https://github.com/fishros/install

All code and models are used under their respective licenses. Please refer to their original repositories for license details.
