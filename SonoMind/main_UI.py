import threading
import queue
import time
import gradio as gr
import json
import re
from langchain_openai import ChatOpenAI
from agents.triage import TriageAgent
from agents.sonographer import SonographerAgent
from agents.radiologist import RadiologistAgent
from agents.physician import PhysicianAgent
import json
import re
from utils import normalize_agent, user_input_queue
from tools.sonographer_tools import SONOGRAPHER_TOOLS
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import threading

reset_event = threading.Event()
chat_history = [] 
sup_input = None

API_KEY = '0'
triage_base_url = "..."
sonographer_base_url =  "..."
radiologist_base_url =  "..."
physician_base_url =  "..."

triage_llm = ChatOpenAI(model="triage_llm", api_key=API_KEY, base_url=triage_base_url)
sonographer_llm = ChatOpenAI(model="sonographer_llm", api_key=API_KEY, base_url=sonographer_base_url) 
radiologist_llm = ChatOpenAI(model="radiologist_llm", api_key=API_KEY, base_url=radiologist_base_url)
physician_llm = ChatOpenAI(model="physician_llm", api_key=API_KEY, base_url=physician_base_url)

RAG_DB_PATH = r"SonoMind/tools/vector_db_docx_pubmedbert-base_structured/chroma"
EMBEDDING_MODEL = r"SonoMind/tools/pubmedbert-base"

current_agent = "supervisor"
execution_history = []
nodule_detection = []
patient_info = ""
patient_ID = "001"
def on_user_input(text: str):
    user_input_queue.put(text)


def safe_json_parse(data):
    if isinstance(data, dict):
        return data

    if not isinstance(data, str):
        raise TypeError(f"Expect dict or str, got {type(data)}")

    cleaned = data.strip()
    cleaned = re.sub(r"```json|```", "", cleaned).strip()
    cleaned = cleaned.replace('"""', "").strip()
    cleaned = cleaned.replace("\n", " ").strip()

    try:
        return json.loads(cleaned)
    except Exception:
        return {"error": "JSON parse failed", "raw": data}

def agent_main_loop(log_callback):
    def ui_logger(msg, agent="system"):
        log_callback(msg, agent=agent)

    supervisor = TriageAgent(triage_llm)
    sonographer = SonographerAgent(tools=SONOGRAPHER_TOOLS, llm=sonographer_llm, logger=ui_logger)
    radiologist = RadiologistAgent(radiologist_llm,logger=ui_logger,patient_ID=patient_ID)
    physician = PhysicianAgent(
        llm=physician_llm,
        rag_db_path=RAG_DB_PATH,
        embedding_model=EMBEDDING_MODEL,
        logger=ui_logger,
        patient_ID=patient_ID
    )

    global current_agent, sup_input, user_input_queue, execution_history, nodule_detection # interrupt_flag, 

    log_callback("===== Agent system started =====", agent="system")

    while True:
        if reset_event.is_set():
            reset_event.clear()
            current_agent = "supervisor"
            user_input_queue = queue.Queue()  
            sup_input = None
            execution_history = []
            nodule_detection = []
            log_callback("===== System Reset: Starting from Supervisor =====", agent="system")
            continue  

        if current_agent == "supervisor":
            if sup_input is not None:
                input_for_supervisor = sup_input
                sup_input = None
            else:
                input_for_supervisor = user_input_queue.get()
                
            sup_output = supervisor.run(input_for_supervisor)
            parsed = safe_json_parse(sup_output)
            log_callback(f"[Supervisor]: {parsed}", agent="supervisor")

            tool_calls = parsed.get("tool_calls", {})
            if not tool_calls:
                continue

            action = tool_calls.get("action")
            agent_name = normalize_agent(tool_calls.get("agent", ""))
            
            if action == "transfer_to_agent" and agent_name:
                log_callback(f"➡ Switching to {agent_name} agent", agent="system")
                if agent_name == "sonographer":
                    patient_info = tool_calls["input"]
                    initial_input = {
                        "patient_info": patient_info,
                        "execution_history": execution_history,
                        "nodule_detection": nodule_detection,
                        "user_input": ""
                    }
                    sup_input = sonographer.loop(initial_input)
                    current_agent = "supervisor"
                    continue

                if agent_name == "radiologist":
                    agent_input = tool_calls["input"]
                    sup_input = radiologist.loop(agent_input)
                    current_agent = "supervisor"
                    continue

                if agent_name == "physician":
                    agent_input = tool_calls["input"]
                    sup_input = physician.run(user_input_queue=user_input_queue)
                    log_callback("Examination finished", agent="system")
                    break

        time.sleep(0.1)

def launch_gradio():
    logs = []
    log_lock = threading.Lock()
    AGENT_COLORS = {
        "supervisor": "rgba(245,245,245, 0.5)",
        "sonographer": "rgba(189,214,201,0.5)",
        "radiologist": "rgba(220,238,254,0.5)",
        "physician": "rgba(254,225,225,0.5)",
        "system": "rgba(255,255,255,0.5)",
        "user": "rgba(237,237,237,0.5)"
    }

    def log_callback(msg, agent="system"):
        with log_lock:
            color = AGENT_COLORS.get(agent, "#DFF2BF")
            logs.append(
                f'<div style="padding:5px; margin:5px 0; border:1px solid transparent; '
                f'border-radius:5px; background-color:{color};">{msg}</div>'
            )

    def submit(text):
        if text.strip():  
            log_callback(f"[User]: {text}", agent="user")
            on_user_input(text)
        return ""

    def refresh():
        with log_lock:
            return "".join(logs)

    def clean():
        with log_lock:
            logs.clear()
        reset_event.set()

    with gr.Blocks() as demo:
        gr.Markdown("## 🧠 IS-MAN: Multi-Agent Ultrasound Assistant")

        output_box = gr.HTML(
            label="Dialogue",
            elem_id="output_box"
        )

        with gr.Row(equal_height=True):
            input_box = gr.Textbox(
                label="User Input",
                placeholder="Type here and press Send",
                scale=4, 
                container=False
            )
            send_btn = gr.Button("Send", scale=1)
            clean_btn = gr.Button("Clean", scale=1)

        send_btn.click(fn=submit, inputs=input_box, outputs=input_box)
        clean_btn.click(fn=clean, inputs=[], outputs=[])
        timer = gr.Timer(0.5)
        timer.tick(refresh, None, output_box)

    threading.Thread(
        target=agent_main_loop,
        args=(log_callback,),
        daemon=True
    ).start()

    demo.launch()




if __name__ == "__main__":
    launch_gradio()
