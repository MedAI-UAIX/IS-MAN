# SonoPilot: Adaptive Force-Position Control & Perception Framework for Robotic Ultrasound Scanning

![SonoPilot System Overview](../demo/demo_sonopilot.jpg)

**SonoPilot** is the core execution module of the IS-MAN (Intelligent Sono-Robotic Multi-Agent Nexus) system. It is designed to autonomously control a robotic manipulator for conducting high-precision ultrasound scanning with compliant and adaptive interaction capabilities. It integrates real-time visual perception modules for closed-loop scanning guidance.

---

## 🔧 Overview of Capabilities

### Robot Control
- Dynamic Controller Switching
- Cartesian Position Control
- Impedance Control
- Hybrid Force-Position Control

### Real-Time Visual Perception
- Multi-Organ Segmentation (thyroid, carotid artery, trachea, liver)
- Thyroid Nodule Detection
- Anatomical Keypoint Localization

### System Interfaces
- Robot Control API Server for SonoMind integration
- Python SDK for programmatic control

---

## Repository Structure

```
SonoPilot/
├── Segmentation/            # 🏥 Real-time Multi-Organ Segmentation
│   ├── model/
│   ├── input/
│   ├── output/
│   ├── Inference.py
│   └── README.md
├── LesionDetection/         # 🔍 Thyroid Nodule Detection
│   ├── input/
│   ├── output/
│   ├── inference.py
│   └── README.md
├── KeypointDetection/       # 🎯 Anatomical Landmark Localization
│   ├── input/
│   ├── output/
│   ├── inference.py
│   └── README.md
├── RobotServer/             # 🔌 Robot Control API Server
├── ForcePosition_calib.py   # ⚙️ Hybrid Force-Position Controller Core
├── SonoPilot.py             # 🚀 Main Scanning Control Interface
├── get_path_intP.py         # 📐 Path Planning Utilities
├── requirements.txt
└── README.md
```

---

## 📌 Dependencies

The version requirements listed below are recommended configurations. Our code may be compatible with other mainstream versions of these dependencies, but full compatibility testing has not been completed.

- [Ubuntu 20.04](https://releases.ubuntu.com/20.04/)
- [ROS Noetic](https://wiki.ros.org/noetic)
- [MoveIt](https://moveit.ai/)
- [Franka ROS Interface](https://github.com/frankarobotics/franka_ros)
- Python ≥3.7
- [libfranka ≥0.13.0](https://github.com/frankarobotics/libfranka)

---

## Installation & Configuration

**Environment Setup**

```bash
conda create -n franka python=3.10
conda activate franka
pip install -r requirements.txt
```

---

## Perception Sub-Modules

### 🏥 Segmentation — Multi-Organ Real-time Segmentation

UNet with reparameterizable convolution for multi-organ segmentation in ultrasound images (thyroid, carotid artery, trachea, liver). Supports both CUDA and CPU inference.

**→ [Segmentation Documentation](Segmentation/README.md)**

---

### 🔍 LesionDetection — Thyroid Nodule Detection

YOLO-based detection model specifically designed for thyroid nodule detection in B-mode ultrasound images. Optimized for real-time clinical deployment.

**→ [LesionDetection Documentation](LesionDetection/README.md)**

---

### 🎯 KeypointDetection — Anatomical Landmark Localization

YOLO-Pose based keypoint detection for thyroid localization. Predicts sparse anatomical keypoints to characterize thyroid pose and morphology for stable robotic alignment.

**→ [KeypointDetection Documentation](KeypointDetection/README.md)**

---

## 📂 Core Files

### [`SonoPilot.py`](SonoPilot.py) — Main Control Interface

The main entry point for the SonoPilot scanning control system. Provides a unified API for SonoMind integration, encapsulating controller management, trajectory execution, and perception feedback loops.

---

### [`ForcePosition_calib.py`](ForcePosition_calib.py) — Hybrid Force-Position Controller

Includes:
- Initialization of Franka Emika Panda robot interface
- Calibration of hybrid force-position control
- Real-time force feedback handling
- Dynamic controller switching logic

---

### [`get_path_intP.py`](get_path_intP.py) — Path Planning Utilities

Trajectory generation and interpolation utilities for smooth scanning path planning.

---

## 🚀 Controller Usage Examples

### Dynamic Controller Switching

Dynamic controller switching is the mechanism that allows SonoPilot to transition smoothly between different control strategies (e.g., from free-space movement to tissue contact).

**Fundamental Rules:**
1. **Check First**: Always query the system for the current status of all controllers.
2. **Match States**:
   - `start_controllers`: **Only accepts controllers currently in the `stopped` state.**
   - `stop_controllers`: **Only accepts controllers currently in the `running` state.**

#### Step 1: Inspect Current States

```python
from ForcePosition_calib import MoveItFranka
franka = MoveItFranka()
franka.list_controllers()
```

Expected Terminal Output:
```python
Controller name: position_force_hybird_controller
Controller state: running
Controller name: franka_state_controller
Controller state: running
Controller name: cartesian_pose_ZJK_controller
Controller state: stopped
```

#### Step 2: Construct the Switch Command

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

### Cartesian Position Controller

```python
from ForcePosition_calib import MoveItFranka
franka = MoveItFranka()
franka.switch_controllers(
    [franka.hybrid_force_position_controller_name],
    [franka.cartesian_position_controller_name]
)
```

---

### Impedance Controller

```python
from ForcePosition_calib import MoveItFranka
franka = MoveItFranka()
franka.hybrid_2_impedance()
franka.update_param(is_scaning=False, translational_stiffness=300, rotational_stiffness=50)
```

---

### Hybrid Force-Position Controller

```python
from ForcePosition_calib import MoveItFranka
franka = MoveItFranka()
franka.impedance_2_hybrid()
franka.update_param(is_scaning=True, ext_force=2)
```

---

## 📖 Controller Parameters Explained

### `update_param()` in Impedance or Hybrid Control

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

### `switch_controllers()`

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

## 📬 Contact

For issues or improvements, please open an Issue.
