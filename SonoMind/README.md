# SonoMind — Multi-Agent Autonomous Ultrasound

## Repository structure

```
sonomind/
│  main_UI.py
│  README.md
│  requirements.txt
│
├─agents
│  │  physician.py
│  │  radiologist.py
│  │  sonographer.py
│  └─triage.py
│
├─fine_tune
│  │  lora_train.py
│  └─data
│
├─models
│  │  physician_agent.yaml
│  │  radiologist_agent.yaml
│  │  run_agents.py
│  │  sonographer_agent.yaml
│  │  triage_agent.yaml
│  │
│  ├─physician_adapter
│  ├─qwen3-4B
│  ├─qwen3-8B
│  ├─qwen3-8B-vl
│  ├─radiologist_adapter
│  └─sonographer_adapter
│
├─prompts
│  │  physician_prompt.py
│  │  radiologist_prompt.py
│  │  sonographer_prompt.py
│  └─ triage_prompt.py
│
└─tools
    │  radiologist_tools.py
    │  sonographer_tools.py
    │  __init__.py
    │
    ├─database
    ├─pubmedbert-base
    ├─thynet
    ├─thynets
    └─thyroid_TIRADS
```

---

## Quickstart (local simulation)

1. Clone the repo and create a virtual environment.

pip install -r requirements.txt


2. Start the tool API server

3. Start the agents

    python SonoMind/models/run_agents.py

4. Run a simple inference (example):

    python main_UI.py


## Model Foundation

Our system is built upon the **Qwen3 family of models**, including:

- **[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)**
- **[Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)**
- **[Qwen3-8B-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)**

These models are trained and accessed using **LLaMAFactory**, following the official implementations and usage guidelines provided by the Qwen team. We sincerely appreciate their contributions to the open-source community.  

- **[LLaMAFactory Repository](https://github.com/hiyouga/LLaMA-Factory/tree/main)**

In building our system, we also leverage **[LangChain](https://github.com/langchain-ai/langchain)** for orchestrat