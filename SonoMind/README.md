# SonoMind — Multi-Agent Autonomous Ultrasound

![Keypoint Overview](demo/../../demo/demo_sonomind.jpg)

## Repository structure

```
sonomind/
├── README.md                 # This file (overview + quickstart)
├── requirements.txt          # minimal Python deps
├── prompts/
│   ├── triage_prompt.py
│   ├── sonographer_prompt.py
│   ├── radiologist_prompt.py
│   └── physician_prompt.py
├── tools/                  
├── agents/
│   ├── triage_agent.py
│   ├── sonographer_agent.py
│   ├── radiologist_agent.py
│   └── physician_agent.py
├── models/
│   ├── prompts.py
│   └── inference.py          # model wrapper/adapter
├── fine_tune/
│   ├── lora_train.py
│   └── data/           
└── ...
```

---

## Quickstart (local simulation)

1. Clone the repo and create a virtual environment.
``` bash
pip install -r requirements.txt
```

2. Start the tool API server

3. Start the agents
``` bash
python SonoMind/models/run_agents.py
```
4. Run a simple inference (example):
``` bash
python agents/triage_agent.py
```


## Model Foundation

Our system is built upon the **Qwen3 family of models**, including:

- **[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)**
- **[Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)**
- **[Qwen3-8B-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)**

These models are trained and accessed using **LLaMAFactory**, following the official implementations and usage guidelines provided by the Qwen team. We sincerely appreciate their contributions to the open-source community.  

- **[LLaMAFactory Repository](https://github.com/hiyouga/LLaMA-Factory/tree/main)**

In building our system, we also leverage **[LangChain](https://github.com/langchain-ai/langchain)** for orchestrat