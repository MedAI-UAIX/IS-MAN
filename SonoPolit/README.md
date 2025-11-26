# SonoPilot: Adaptive Force-Position Control Framework for Robotic Ultrasound Scanning

![SonoPilot System Overview](SonoPilot.jpg)

**SonoPilot** is a core module of the IS-MAN (Intelligent Sono-Robotic Multi-Agent Nexus) system. It is designed to autonomously control a robotic manipulator for conducting high-precision ultrasound scanning with compliant and adaptive interaction capabilities.

---

## 🔧 Overview of Capabilities

This system enables seamless switching between:

- Cartesian Position Control
- Impedance Control
- Hybrid Force-Position Control
- Dynamic Controller Switching

---

## 📂 Key File: `ForcePosition_calib.py`

Includes:

- Initialization of Franka Emika Panda robot interface
- Calibration of hybrid force-position control
- Real-time force feedback handling

---

## 🚀 Controller Usage Examples

### Cartesian Position Controller

```python
franka = MoveItFranka()
franka.switch_controllers(
    [franka.hybrid_force_position_controller_name],
    [franka.cartesian_position_controller_name]
)
```

### Impedance Controller

```python
franka = MoveItFranka()
franka.hybrid_2_impedance()
franka.update_param(is_scaning=False, translational_stiffness=300, rotational_stiffness=50)
```

### Hybrid Force-Position Controller

```python
franka = MoveItFranka()
franka.impedance_2_hybrid()
franka.update_param(is_scaning=True, ext_force=2)
```

### Dynamic Controller Switching

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

## 📌 Dependencies

- ROS Noetic
- MoveIt
- Franka ROS Interface
- Python ≥3.7

---

## 📚 Citation

If you use this work, please cite our IS-MAN project paper on robotic thyroid ultrasound screening.
