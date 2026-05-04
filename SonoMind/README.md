# SonoMind — Multi-Agent Autonomous Ultrasound


![SonoMind System Overview](https://github.com/MedAI-UAIX/IS-MAN/blob/main/demo/demo_sonomind.jpg)

## Repository structure

```
.
├── agents
│   ├── orchestrator.py
│   ├── physician.py
│   ├── radiologist.py
│   └── sonographer.py
├── main_UI.py
├── models
│   ├── orchestrator.yaml
│   ├── physician.yaml
│   ├── radiologist.yaml
│   ├── run_agents.py
│   └── sonographer.yaml
├── patient
│   └── 001
├── prompt
│   ├── orchestrator_prompt.py
│   ├── physician_prompt.py
│   ├── radiologist_prompt.py
│   └── sonographer_prompt.py
├── README.md
├── requirements.txt
├── tools
│   ├── build_vector_db.py
│   ├── FrankaRequest.py
│   ├── pubmedbert-base
│   ├── radiologist_tools.py
│   ├── RobotServer
│   ├── SonoPilot.py
│   ├── throid_TI-RADS
│   ├── thynet
│   ├── thynets
│   └── vector_db_docx_pubmedbert-base_structured
└── utils.py
```

---

## Model Configuration

SonoMind supports different large language models through API-based model services. Users may run the framework with commercial LLM/VLM providers, locally deployed open-source models, or the model weights used in this study, which are available upon reasonable request.

### 1. Using Commercial API Providers

To use commercial API providers, obtain an API key and base URL from at least one supported provider, such as:

- [OpenAI](https://platform.openai.com/)
- [Anthropic Claude](https://console.anthropic.com/)
- [Google Gemini](https://ai.google.dev/)
- [DeepSeek](https://platform.deepseek.com/)

Then specify the key and endpoint URLs in `main_UI.py`:

```python
API_KEY = "your_api_key"

orchestrator_base_url = "your_orchestrator_base_url"
sonographer_base_url = "your_sonographer_base_url"
radiologist_base_url = "your_radiologist_base_url"
physician_base_url = "your_physician_base_url"
```

Each agent can be configured with a different model service. 

---

### 2. Using Local Models

In this study, local large language models were deployed as API services using [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main).

#### Step 1. Install LLaMA-Factory

```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory

conda create -n llamafactory python=3.10
conda activate llamafactory

pip install -e .
pip install -r requirements/metrics.txt
```

#### Step 2. Download Model Weights

Download the required model weights and place them under:

```bash
SonoMind/models/
```

The models used in this study include:

- **[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)**
- **[Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)**
- **[Qwen3-8B-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)**

Model backbones can be downloaded from the corresponding model repositories. Fine-tuned adapters used in this study can be configured through `adapter_name_or_path` in the corresponding YAML files and may be made available for academic use upon reasonable request by contacting the corresponding author.

#### Step 3. Configure Model YAML Files

Configure the model YAML files in:

```bash
SonoMind/models/
```

> **Note:** For functional testing or custom deployment without fine-tuned adapters, `adapter_name_or_path` can be commented out.

Each YAML file should specify the base model path, optional adapter path, prompt template, and inference backend.

#### Step 4. Start Local Model API Services

Run:

```bash
python SonoMind/models/run_agents.py
```

This script launches the configured local models as API services.

#### Step 5. Update API URLs

Copy the generated API URLs into `main_UI.py`:

```python
API_KEY = "0"  # default value for local model services

orchestrator_base_url = "your_orchestrator_base_url"
sonographer_base_url = "your_sonographer_base_url"
radiologist_base_url = "your_radiologist_base_url"
physician_base_url = "your_physician_base_url"
```

---

## Tools Configuration

SonoMind uses external tools for robotic scanning, image interpretation, and clinical decision support. Please configure the required tools before running the full system.

### 1. Sonographer Agent Tools

The sonographer agent interacts with the robotic arm control API. Start the robotic arm control service using the environment configured in the SonoPilot module:

```bash
conda activate franka
python SonoPilot.py
```

**If no robotic arm is available, you can still verify the code by following the [Quickstart](#Quickstart) guide. In the user interface, locate the "Robot Execution" status panel and click the "Simulate" button as shown in the image below.**

![demo_sonomindUI](https://github.com/MedAI-UAIX/IS-MAN/blob/main/demo/demo_sonomindUI.png)

---

### 2. Radiologist Agent Tools

The radiologist agent uses additional tools for thyroid ultrasound image analysis and follow-up recommendation generation.

#### Follow-up Tool

The follow-up LLM tool was developed based on [Gemini 2.5 Flash](https://ai.google.dev/gemini-api/docs/models#gemini-2.5-flash). To use this tool, obtain an API key and endpoint URL from Google AI Studio, and specify them in the `FollowUpTool` configuration in `tools/radiologist_tools.py`.

#### Diagnostic Models

The default weights are used when running the code directly. The weights used for thyroid nodule benign–malignant diagnosis, including **ThyNet** and **ThyNet-S**, as well as the model used for **TI-RADS classification**, may be made available for academic use upon reasonable request.

---

### 3. Physician Agent Tools

The physician agent uses PubMedBERT-indexed guideline knowledge to support evidence-informed clinical recommendations.

#### Model Preparation
Before running the physician agent, **please manually download the pre-trained PubMedBERT model** from [Hugging Face](https://huggingface.co/NeuML/pubmedbert-base-embeddings/tree/main) and place it into the [`SonoMind/tools/`](https://github.com/MedAI-UAIX/IS-MAN/tree/main/SonoMind/tools) folder.

---

## Quickstart
We provide a demo to test the feasibility of the SonoMind workflow.

### 1. Clone the Repository and Create a Virtual Environment

```bash
cd SonoMind

conda create -n sonomind python=3.10
conda activate sonomind
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Models and Start the Tool API Servers

Download the required model weights and configure the model files as described in [Model Configuration](#model-configuration).

If you use the robotic arm, also start the sonographer tool API as described in [Tools Configuration](#tools-configuration).

### 4. Start the Agent Services

```bash
python SonoMind/models/run_agents.py
```

> **Note:** If your local machine has limited computational resources, you can deploy the agents on a remote server. However, to minimize latency and ensure the safety of the robotic arm, it is strongly recommended to deploy the sonographer agent on the same machine that directly controls the robotic arm.

### 5. Specify the Patient ID

Specify the patient ID in main_UI.py. The default patient ID is 001.

### 5. Run a Simple Inference Demo

```bash
python main_UI.py
```

---

## Demo Video

A demonstration video of the SonoMind workflow is available. You can refer to this video for a quick overview of the system's capabilities and usage.



https://github.com/user-attachments/assets/8aa79891-08ea-4dae-ab16-3343e0fedeaf



---

## Model Foundation

Our system is built upon the **Qwen3 family of models**, including:

- **[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)**
- **[Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)**
- **[Qwen3-8B-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)**

These models are trained and accessed using **LLaMAFactory**, following the official implementations and usage guidelines provided by the Qwen team. We sincerely appreciate their contributions to the open-source community.  

- **[LLaMAFactory Repository](https://github.com/hiyouga/LLaMA-Factory/tree/main)**

In building our system, we also leverage **[LangChain](https://github.com/langchain-ai/langchain)** for orchestration.
