# IS-MAN: LLM-driven Embodied Intelligence System for Autonomous Ultrasound

![IS-MANt System Overview](IS-MAN.jpg)

> This repository contains the source code, and pretrained models for the IS-MAN system, a large-scale LLM-empowered embodied intelligence framework for fully autonomous ultrasound-based thyroid cancer screening in real-world clinical settings.

## Overview

**IS-MAN** (Intelligent Sono-Robotic Multi-Agent Nexus) is a clinically validated embodied AI system integrating robotic force control, dynamic visual servoing, and LLM-powered multi-agent decision-making for autonomous ultrasound-based thyroid screening. It comprises two synergistic modules:

- **SonoPilot**: A real-time adaptive control module responsible for safe, stable, and anatomically precise robotic scanning.
- **SonoMind**: A cognitive multi-agent system orchestrated by a large language model, enabling perception, decision-making, and clinical reasoning across four roles—triage, sonographer, radiologist, and physician.

IS-MAN was deployed and evaluated in a prospective human study with over 500 participants, achieving high scanning success rates, clinically equivalent image quality, and expert-level diagnosis accuracy, marking a significant advancement in autonomous medical robotics.

---

## Repository Structure

```
IS-MAN
├── Sonopilot
├── Sonomind
├── SegNet
├── LesionDetection
├── KeypointDetection
└── Franka_ws
```

- **SonoPilot**: Implementations of hybrid force–position control, impedance control, and Cartesian position control for the robotic manipulator.
- **SonoMind**: The multi-agent framework coordinating perception, control, and decision modules as used in the autonomous workflow.
- **SonoMind**: Real-time segmentation: a segmentation pipeline supporting multiple target anatomies (thyroid, carotid artery, liver) with inference scripts and model configurations for clinical deployment.
- **LesionDetection**: Thyroid-nodule detection module with training/inference scripts and evaluation utilities
- **KeypointDetection**: Keypoint Detection for Thyroid Localization
- **Franka_ws**: 

---

## Installation & Configuration

1. **Environment Setup**

```bash
conda create -n isman python=3.10
conda activate isman
pip install -r requirements.txt
```

Dependencies include:

- PyTorch >= 2.0
- OpenCV
- Transformers (HuggingFace)
- ROS (for real-time robot control)
- Gradio / Streamlit (for UI demo)

2. **Pretrained Models**

```bash
bash scripts/download_models.sh
```

3. **Hardware Requirements**

- 7-DOF robotic arm with force-torque sensor
- Real-time ultrasound imaging system (DICOM compatible)
- GPU with >12GB memory (for segmentation)

---

## Running the System

```bash
roslaunch sonopilot_control.launch
python sonopilot/servoing/main.py
python sonomind/agent_orchestration.py --mode inference
python scripts/demo_patient_pipeline.py --input patient_xyz
```


## License & Data Availability

All source code is made available under the **MIT License**. The datasets used in our experiments are **de-identified** and released under a CC BY-NC 4.0 license for non-commercial academic use.

---

### Compliance with Nature’s Open Science Policy

- The data used for training and testing are included in the repository in a processed, anonymized format.
- All pretrained models and prompts used in SonoMind are made publicly accessible.
- Detailed installation and execution instructions are provided to enable full reproducibility.

