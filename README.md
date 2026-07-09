# LLM-driven embodied intelligence for workflow-level autonomous robotic ultrasound: a dual-center prospective randomized crossover study

![IS-MAN System Overview](demo/IS-MAN.jpg)

> This repository contains the source code, and pretrained models for the IS-MAN system, a large-scale LLM-empowered embodied intelligence framework for fully autonomous ultrasound-based thyroid cancer screening in real-world clinical settings.

---

## Overview

**IS-MAN** (Intelligent Sono-Robotic Multi-Agent Nexus) is a clinically validated embodied AI system integrating robotic force control, dynamic visual servoing, and LLM-powered multi-agent decision-making for autonomous ultrasound-based thyroid screening. It comprises two synergistic core modules:

- **SonoPilot**: A real-time adaptive control module responsible for safe, stable, and anatomically precise robotic scanning. It integrates force-position hybrid control, impedance control, and real-time visual perception (segmentation, lesion detection, keypoint localization) to enable closed-loop robotic ultrasound scanning.

- **SonoMind**: A cognitive multi-agent system orchestrated by a large language model, enabling perception, decision-making, and clinical reasoning across four roles—orchestrator, sonographer, radiologist, and physician.

IS-MAN was deployed and evaluated in a prospective human study with over 500 participants, achieving high scanning success rates, clinically equivalent image quality, and expert-level diagnosis accuracy, marking a significant advancement in autonomous medical robotics.

---

## Repository Structure

```
IS-MAN/
├── SonoPilot                  # 🤖 Autonomous Scanning Control & Perception
│   ├── Segmentation           # 🏥 Real-time Multi-Organ Segmentation
│   ├── LesionDetection        # 🔍 Thyroid Nodule Detection
│   ├── KeypointDetection      # 🎯 Anatomical Landmark Localization
│   ├── RobotServer            # 🔌 Robot Control API Server
│   ├── ForcePosition_calib.py # ⚙️ Hybrid Force-Position Controller
│   ├── SonoPilot.py           # 🚀 Main Scanning Control Interface
│   ├── get_path_intP.py       # 📐 Path Planning Utilities
│   ├── requirements.txt
│   └── README.md
├── SonoMind                   # 🧠 LLM-Powered Multi-Agent Cognition
│   ├── agents                 # 👥 Four Role Agents (Orchestrator/Sonographer/Radiologist/Physician)
│   ├── models                 # 📦 Model Configuration & API Launcher
│   ├── prompt                 # 📝 Agent Prompt Templates
│   ├── tools                  # 🔧 Perception & Control Toolkits
│   ├── patient                # 📋 Patient Data Directory
│   ├── main_UI.py             # 🖥️ Main User Interface
│   ├── sonopilot2sonomind.py  # 🔗 SonoPilot ↔ SonoMind Bridge
│   ├── utils.py
│   ├── requirements.txt
│   └── README.md
├── Franka_ws                  # 🛠️ ROS Development Workspace
│   ├── src
│   └── README.md
├── demo                       # 🎥 System Demonstration Assets
├── LICENSE
└── README.md
```

---

## Module Overview

### 🛠️ Franka_ws — ROS Development Workspace

The ROS workspace designed for developing and running control applications for the [Franka Emika Panda](https://franka.de/) robot. It integrates the Franka ROS packages and provides a clean environment for real-time robot control, motion execution, and controller development.

**→ [Franka_ws Detailed Documentation](Franka_ws/README.md)**

---

### 🤖 SonoPilot — Adaptive Force-Position Control & Perception Framework

SonoPilot is the core execution module of IS-MAN, designed to autonomously control a robotic manipulator for high-precision ultrasound scanning with compliant and adaptive interaction capabilities. It integrates real-time visual perception modules for closed-loop scanning guidance.

**Key Capabilities:**
- Dynamic Controller Switching (Position / Impedance / Hybrid Force-Position)
- Real-time Multi-Organ Segmentation (thyroid, carotid artery, trachea, liver)
- Thyroid Nodule Detection
- Anatomical Keypoint Localization
- Robot Control API Server

**→ [SonoPilot Detailed Documentation](SonoPilot/README.md)**

---

### 🧠 SonoMind — Multi-Agent Autonomous Ultrasound System

SonoMind is the LLM-orchestrated multi-agent cognitive framework that coordinates the entire autonomous ultrasound workflow, from patient triage to final clinical recommendations.

**Four Agent Roles:**
- **Orchestrator**: Workflow planning and task dispatching
- **Sonographer**: Robotic scanning control & image acquisition guidance
- **Radiologist**: Image interpretation, nodule analysis & TI-RADS classification
- **Physician**: Evidence-based clinical recommendations & follow-up advice

**→ [SonoMind Detailed Documentation](SonoMind/README.md)**

---

### 🎥 demo — Demonstration Assets

Contains sample images, architecture diagrams, and UI screenshots demonstrating system capabilities across all modules.

---

## Installation

### Prerequisites

- Ubuntu 20.04
- ROS Noetic
- Python ≥ 3.10
- Conda (recommended for environment management)

### Setup

#### 1. **Clone the repository**

```bash
git clone https://github.com/MedAI-UAIX/IS-MAN.git
cd IS-MAN/Franka_ws
```

#### 2. **Build the workspace**

```bash
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
```

#### 3. **Source ROS workspace**

```bash
source devel/setup.bash
```

#### 4. **Start ROS Master**

Open the first terminal and start the ROS core:
```bash
roscore
```

#### 5. **Launch Position-Force Controller**

Open a second terminal and launch the controller:
```bash
roslaunch franka_example_controllers position_force_example.launch
```

#### 6. Install Dependencies (SonoPilot)

cd ../SonoPilot
pip install -r requirements.txt

#### 7. Install Dependencies (SonoMind)

cd ../SonoMind
pip install -r requirements.txt
#### Download Qwen3 models (Configure path in models/config.yaml)

#### 8. Run the System

**Start the Robot Control Server (SonoPilot):**
```bash
conda activate franka
cd SonoPilot
python SonoPilot.py
```

**Start the Multi-Agent System (SonoMind):**
```bash
conda activate sonomind
cd SonoMind
python models/run_agents.py  # Start model API services
python main_UI.py            # Launch the main interface
```

> 💡 **Simulation Mode**: If no robotic arm is available, you can still verify the workflow by clicking the "Simulate" button in the Robot Execution status panel of the UI.

---

## License

All source code is made available under the **Apache-2.0 License**.

---

## 🔗 Dependencies and Acknowledgements

This project builds upon several open-source projects and pretrained models. We gratefully acknowledge the following repositories and frameworks:

- **libfranka** for low-level robotic control
  https://github.com/frankaemika/libfranka
- **Qwen3 Models** for LLM-based agents in SonoMind
  https://huggingface.co/Qwen
- **LLaMA-Factory** for model deployment and fine-tuning
  https://github.com/hiyouga/LLaMA-Factory/tree/main
- **LangChain** for agent orchestration
  https://github.com/langchain-ai/langchain
- **Ultralytics YOLO** for detection and keypoint models
  https://github.com/ultralytics/ultralytics
- **PubMedBERT** for medical text embeddings
  https://huggingface.co/NeuML/pubmedbert-base-embeddings

All code and models are used under their respective licenses. Please refer to their original repositories for license details.
