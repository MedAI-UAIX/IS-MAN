import json
import re
import time
import queue
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import gradio as gr
import requests
from langchain_openai import ChatOpenAI

from agents.orchestrator import OrchestratorAgent
from agents.sonographer import SonographerAgent
from agents.radiologist import RadiologistAgent
from agents.physician import PhysicianAgent

from utils import normalize_agent
import warnings
warnings.filterwarnings("ignore")


# model config

save_root = './patient'
patient_id = "001"
API_KEY = 'your_api_key'

orchestrator_base_url = "your_orchestrator_base_url"
sonographer_base_url = "your_sonographer_base_url"
radiologist_base_url = "your_radiologist_base_url"
physician_base_url = "your_physician_base_url"

orchestrator_llm = ChatOpenAI(
    model="orchestrator_llm",
    api_key=API_KEY,
    base_url=orchestrator_base_url,
    timeout=30,
    max_retries=3,
)
sonographer_llm = ChatOpenAI(
    model="sonographer_llm",
    api_key=API_KEY,
    base_url=sonographer_base_url,
    timeout=30,
    max_retries=3,
)
radiologist_llm = ChatOpenAI(
    model="radiologist_llm",
    api_key=API_KEY,
    base_url=radiologist_base_url,
    timeout=30,
    max_retries=3,
)
physician_llm = ChatOpenAI(
    model="physician_llm",
    api_key=API_KEY,
    base_url=physician_base_url,
    timeout=30,
    max_retries=3,
)

RAG_DB_PATH = "./tools/vector_db_docx_pubmedbert-base_structured/chroma"
EMBEDDING_MODEL = "./tools/pubmedbert-base"



SERVICE_LOCK = threading.Lock()
SERVICE_STATUS = {
    "orchestrator": {"ok": False, "msg": "Not checked"},
    "sonographer": {"ok": False, "msg": "Not checked"},
    "radiologist": {"ok": False, "msg": "Not checked"},
    "physician": {"ok": False, "msg": "Not checked"},
}

SERVICE_URLS = {
    "orchestrator": orchestrator_base_url,
    "sonographer": sonographer_base_url,
    "radiologist": radiologist_base_url,
    "physician": physician_base_url,
}


def normalize_base_url(base_url: str) -> str:
    return str(base_url).rstrip("/")


def build_healthcheck_urls(base_url: str):
    base_url = normalize_base_url(base_url)

    if base_url.endswith("/v1"):
        root = base_url[:-3].rstrip("/")
        return [
            f"{base_url}/models",
            f"{root}/health",
            f"{root}/models",
        ]
    else:
        return [
            f"{base_url}/v1/models",
            f"{base_url}/health",
            f"{base_url}/models",
        ]


def check_service(base_url: str, api_key: str = "", timeout: float = 2.5):
    if not base_url:
        return False, "Empty base_url"

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    urls = build_healthcheck_urls(base_url)
    last_err = None

    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=timeout)

            if r.status_code == 200:
                return True, f"OK ({url})"
            elif r.status_code in (401, 403):
                return True, f"Reachable but auth failed ({url}, HTTP {r.status_code})"
            else:
                last_err = f"{url} -> HTTP {r.status_code}"

        except requests.exceptions.ConnectTimeout:
            last_err = f"{url} -> connect timeout"
        except requests.exceptions.ReadTimeout:
            last_err = f"{url} -> read timeout"
        except requests.exceptions.ConnectionError as e:
            last_err = f"{url} -> connection error: {e}"
        except Exception as e:
            last_err = f"{url} -> {e}"

    return False, last_err or "Unknown error"


def initial_service_check(log_callback=None):
    for name, url in SERVICE_URLS.items():
        ok, msg = check_service(url, api_key=API_KEY, timeout=2.5)

        with SERVICE_LOCK:
            SERVICE_STATUS[name] = {"ok": ok, "msg": msg}

        if log_callback:
            if ok:
                log_callback(
                    f"[Service Check] {name} service is available: {msg}",
                    agent="system",
                    session_id=SESSION_MANAGER.get().session_id
                )
            else:
                log_callback(
                    f"[Service Check] {name} service is unavailable: {msg}",
                    agent="system",
                    session_id=SESSION_MANAGER.get().session_id
                )


def get_service_status(name: str):
    with SERVICE_LOCK:
        return SERVICE_STATUS.get(name, {"ok": False, "msg": "Unknown service"})



def safe_json_parse(data):
    if isinstance(data, dict):
        return data

    if not isinstance(data, str):
        return {"error": f"Expect dict or str, got {type(data)}", "raw": str(data)}

    cleaned = data.strip()
    cleaned = re.sub(r"```json|```", "", cleaned).strip()
    cleaned = cleaned.replace('"""', "").strip()
    cleaned = cleaned.replace("\n", " ").strip()

    try:
        return json.loads(cleaned)
    except Exception:
        return {"error": "JSON parse failed", "raw": data}


def format_exception(e: Exception) -> str:
    return "".join(traceback.format_exception_only(type(e), e)).strip()


def run_with_timeout_polling(
    func,
    kwargs: Dict[str, Any],
    is_session_valid,
    poll_interval: float = 0.1,
):
    result_queue = queue.Queue()

    def _target():
        try:
            result = func(**kwargs)
            result_queue.put(("ok", result))
        except Exception as e:
            result_queue.put(("err", e))

    t = threading.Thread(target=_target, daemon=True)
    t.start()

    while True:
        if not is_session_valid():
            return {"cancelled": True, "reason": "session_invalidated"}

        try:
            status, payload = result_queue.get(timeout=poll_interval)
            if status == "ok":
                return {"cancelled": False, "result": payload}
            raise payload
        except queue.Empty:
            continue


@dataclass
class SessionState:
    session_id: int
    patient_id: str = patient_id
    input_queue: queue.Queue = field(default_factory=queue.Queue)

    sup_input: Optional[Any] = None
    execution_history: List[Any] = field(default_factory=list)
    nodule_detection: List[Any] = field(default_factory=list)
    patient_info: str = ""
    conversation_history: List[dict] = field(default_factory=list)

    current_stage: str = "Idle"
    current_agent: str = "orchestrator"
    exam_status: str = "Waiting for user input"
    last_error: str = ""

    sonographer_debug: bool = False
    is_active: bool = True


class SessionManager:
    def __init__(self):
        self.lock = threading.Lock()
        self._session_counter = 0
        self.current_session = self._new_session_locked()

    def _new_session_locked(self):
        self._session_counter += 1
        return SessionState(session_id=self._session_counter)

    def reset(self):
        with self.lock:
            if self.current_session:
                self.current_session.is_active = False
            self.current_session = self._new_session_locked()
            return self.current_session

    def get(self) -> SessionState:
        with self.lock:
            return self.current_session

    def is_valid(self, session_id: int) -> bool:
        with self.lock:
            return (
                self.current_session is not None
                and self.current_session.session_id == session_id
                and self.current_session.is_active
            )


SESSION_MANAGER = SessionManager()


def agent_main_loop(log_callback, status_callback):
    log_callback("===== Agent system started =====", agent="system")

    while True:
        session = SESSION_MANAGER.get()

        if not SESSION_MANAGER.is_valid(session.session_id):
            time.sleep(0.05)
            continue

        session_id = session.session_id

        supervisor = OrchestratorAgent(orchestrator_llm)
        radiologist = RadiologistAgent(
            radiologist_llm,
            logger=lambda msg, agent="radiologist", sid=session_id: log_callback(
                msg, agent=agent, session_id=sid
            ),
            patient_ID=patient_id,
        )
        physician = PhysicianAgent(
            llm=physician_llm,
            rag_db_path=RAG_DB_PATH,
            embedding_model=EMBEDDING_MODEL,
            logger=lambda msg, agent="physician", sid=session_id: log_callback(
                msg, agent=agent, session_id=sid
            ),
            patient_ID=patient_id,
        )

        def set_status(stage=None, agent=None, exam_status=None, error=None):
            if not SESSION_MANAGER.is_valid(session.session_id):
                return

            if stage is not None:
                session.current_stage = stage
            if agent is not None:
                session.current_agent = agent
            if exam_status is not None:
                session.exam_status = exam_status
            if error is not None:
                session.last_error = error

            status_callback(session)

        try:
            set_status(
                stage="orchestrator routing",
                agent="orchestrator",
                exam_status="Waiting for orchestrator input",
                error=""
            )

            if session.sup_input is not None:
                input_for_supervisor = session.sup_input
                session.sup_input = None
            else:
                try:
                    input_for_supervisor = session.input_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

            if not SESSION_MANAGER.is_valid(session.session_id):
                continue

            svc = get_service_status("orchestrator")
            if not svc["ok"]:
                set_status(
                    stage="Error",
                    agent="orchestrator",
                    exam_status=f"orchestrator unavailable: {svc['msg']}",
                    error=svc["msg"]
                )
                time.sleep(0.5)
                continue

            set_status(
                stage="orchestrator reasoning",
                agent="orchestrator",
                exam_status="Orchestrator is analyzing input"
            )

            sup_output = supervisor.run(
                input_for_supervisor,
                conversation_history=session.conversation_history[-10:]
            )
            parsed = safe_json_parse(sup_output)

            if parsed.get("error") == "JSON parse failed":
                log_callback(
                    f"[Orchestrator Parse Error] raw output: {parsed.get('raw', '')}",
                    agent="system",
                    session_id=session.session_id
                )

            log_callback(parsed, agent="orchestrator", session_id=session.session_id)

            tool_calls = parsed.get("tool_calls", {})
            if not tool_calls:
                set_status(
                    stage="Idle",
                    agent="orchestrator",
                    exam_status="No tool call returned by orchestrator"
                )
                continue

            action = tool_calls.get("action")
            agent_name = normalize_agent(tool_calls.get("agent", ""))

            session.conversation_history.append({
                "role": "orchestrator",
                "content": f"Decision: action={action}, agent={agent_name}, input={tool_calls.get('input', '')}"
            })

            if action != "transfer_to_agent" or not agent_name:
                set_status(
                    stage="Idle",
                    agent="orchestrator",
                    exam_status=f"Unsupported action: {action}"
                )
                continue

            if agent_name == "sonographer":
                svc = get_service_status("sonographer")
                if not svc["ok"]:
                    log_callback(
                        f"Sonographer service unavailable: {svc['msg']}",
                        agent="system",
                        session_id=session.session_id
                    )
                    set_status(
                        stage="Error",
                        agent="sonographer",
                        exam_status=f"Sonographer unavailable: {svc['msg']}",
                        error=svc["msg"]
                    )
                    continue
                sonographer = SonographerAgent(
                    llm=sonographer_llm,
                    debug_mode=session.sonographer_debug,
                    logger=lambda msg, agent="sonographer", sid=session_id: log_callback(
                        msg, agent=agent, session_id=sid
                    )
                )

                session.patient_info = tool_calls.get("input", "")
                initial_input = {
                    "patient_info": session.patient_info,
                    "execution_history": session.execution_history,
                    "nodule_detection": session.nodule_detection,
                    "user_input": ""
                }

                log_callback(
                    f"Switching to sonographer agent ({'SIMULATED' if session.sonographer_debug else 'CONNECTED'} mode)",
                    agent="system",
                    session_id=session.session_id
                )
                set_status(
                    stage="Ultrasound examination",
                    agent="sonographer",
                    exam_status=f"Sonographer is running ({'SIMULATED' if session.sonographer_debug else 'CONNECTED'} mode)"
                )

                wrapped = run_with_timeout_polling(
                    func=sonographer.loop,
                    kwargs={
                        "initial_input": initial_input,
                        "user_input_queue": session.input_queue,
                    },
                    is_session_valid=lambda sid=session.session_id: SESSION_MANAGER.is_valid(sid),
                )

                if wrapped["cancelled"]:
                    log_callback(
                        "Sonographer result discarded because session was reset.",
                        agent="system",
                        session_id=session.session_id
                    )
                    continue

                session.sup_input = wrapped["result"]
                session.conversation_history.append({
                    "role": "sonographer",
                    "content": wrapped["result"].get("input", str(wrapped["result"]))
                })

                set_status(
                    stage="orchestrator routing",
                    agent="orchestrator",
                    exam_status="Sonographer finished, returning to orchestrator"
                )
                continue

            if agent_name == "radiologist":
                svc = get_service_status("radiologist")
                if not svc["ok"]:
                    log_callback(
                        f"Radiologist service unavailable: {svc['msg']}",
                        agent="system",
                        session_id=session.session_id
                    )
                    set_status(
                        stage="Error",
                        agent="radiologist",
                        exam_status=f"Radiologist unavailable: {svc['msg']}",
                        error=svc["msg"]
                    )
                    continue

                agent_input = tool_calls.get("input", "")

                log_callback(
                    "Switching to radiologist agent",
                    agent="system",
                    session_id=session.session_id
                )
                set_status(
                    stage="Image interpretation",
                    agent="radiologist",
                    exam_status="Radiologist is analyzing images"
                )

                wrapped = run_with_timeout_polling(
                    func=radiologist.loop,
                    kwargs={
                        "user_input": agent_input,
                        "user_input_queue": session.input_queue,
                    },
                    is_session_valid=lambda sid=session.session_id: SESSION_MANAGER.is_valid(sid),
                )

                if wrapped["cancelled"]:
                    log_callback(
                        "Radiologist result discarded because session was reset.",
                        agent="system",
                        session_id=session.session_id
                    )
                    continue

                session.sup_input = wrapped["result"]
                session.conversation_history.append({
                    "role": "radiologist",
                    "content": wrapped["result"].get("input", str(wrapped["result"]))
                })

                set_status(
                    stage="orchestrator routing",
                    agent="orchestrator",
                    exam_status="Radiologist finished, returning to orchestrator"
                )
                continue

            if agent_name == "physician":
                svc = get_service_status("physician")
                if not svc["ok"]:
                    log_callback(
                        f"Physician service unavailable: {svc['msg']}",
                        agent="system",
                        session_id=session.session_id
                    )
                    set_status(
                        stage="Error",
                        agent="physician",
                        exam_status=f"Physician unavailable: {svc['msg']}",
                        error=svc["msg"]
                    )
                    continue

                agent_input = tool_calls.get("input", "")

                log_callback(
                    "Switching to physician agent",
                    agent="system",
                    session_id=session.session_id
                )
                set_status(
                    stage="Clinical recommendation",
                    agent="physician",
                    exam_status="Physician is generating recommendation"
                )

                physician_kwargs = {}
                physician_vars = physician.run.__code__.co_varnames

                if "agent_input" in physician_vars:
                    physician_kwargs["agent_input"] = agent_input
                elif "input_data" in physician_vars:
                    physician_kwargs["input_data"] = agent_input
                elif "data" in physician_vars:
                    physician_kwargs["data"] = agent_input
                elif "user_input" in physician_vars:
                    physician_kwargs["user_input"] = agent_input

                if "user_input_queue" in physician_vars:
                    physician_kwargs["user_input_queue"] = session.input_queue

                wrapped = run_with_timeout_polling(
                    func=physician.run,
                    kwargs=physician_kwargs,
                    is_session_valid=lambda sid=session.session_id: SESSION_MANAGER.is_valid(sid),
                )

                if wrapped["cancelled"]:
                    log_callback(
                        "Physician result discarded because session was reset.",
                        agent="system",
                        session_id=session.session_id
                    )
                    continue

                session.conversation_history.append({
                    "role": "physician",
                    "content": wrapped["result"].get("content", str(wrapped["result"]))
                })

                log_callback("Examination finished", agent="system", session_id=session.session_id)
                log_callback("A new message will start a new session.", agent="system", session_id=session.session_id)
                set_status(
                    stage="Completed",
                    agent="physician",
                    exam_status="Examination finished"
                )
                continue

            set_status(
                stage="Idle",
                agent="orchestrator",
                exam_status=f"Unknown agent: {agent_name}"
            )
            log_callback(
                f"Unknown agent requested by orchestrator: {agent_name}",
                agent="system",
                session_id=session.session_id
            )

        except Exception as e:
            err_msg = format_exception(e)
            tb = traceback.format_exc()

            set_status(
                stage="Error",
                agent="system",
                exam_status="An exception occurred in orchestrator",
                error=err_msg
            )
            log_callback(f"[System Error] {err_msg}", agent="system", session_id=session.session_id)
            log_callback(
                f"<pre style='white-space:pre-wrap'>{tb}</pre>",
                agent="system",
                session_id=session.session_id
            )
            time.sleep(0.2)


# Gradio UI
def launch_gradio():
    logs: List[str] = []
    log_lock = threading.Lock()
    status_lock = threading.Lock()

    latest_status = {
        "session_id": SESSION_MANAGER.get().session_id,
        "patient_id": SESSION_MANAGER.get().patient_id,
        "current_stage": "Idle",
        "current_agent": "orchestrator",
        "exam_status": "Waiting for user input",
        "last_error": "",
        "sonographer_debug": SESSION_MANAGER.get().sonographer_debug,
    }

    AGENT_STYLES = {
        "orchestrator": {"bg": "#F4F7FB", "border": "#D6E0F0", "label": "Orchestrator"},
        "sonographer": {"bg": "#ECF9F1", "border": "#BFE5CC", "label": "Sonographer"},
        "radiologist": {"bg": "#EEF6FF", "border": "#BDD9FF", "label": "Radiologist"},
        "physician": {"bg": "#FFF2F2", "border": "#F2C3C3", "label": "Physician"},
        "system": {"bg": "#FFFFFF", "border": "#E5E7EB", "label": "⚙️ System"},
        "user": {"bg": "#F3F4F6", "border": "#D1D5DB", "label": "👤 User"},
    }

    def log_callback(msg, agent="system", session_id=None):
        with log_lock:
            if session_id is not None and not SESSION_MANAGER.is_valid(session_id):
                return

            style = AGENT_STYLES.get(agent, AGENT_STYLES["system"])
            row_class = "right" if agent == "user" else "left"

            raw_msg = str(msg)

            if agent == "system" and (
                "Switching to" in raw_msg
                or "Examination finished" in raw_msg
                or "Reset" in raw_msg
                or "A new message will start a new session." in raw_msg
            ):
                logs.append(f'<div class="timeline-event">{raw_msg}</div>')
                return

            logs.append(
                f"""
                <div class="chat-row {row_class}">
                    <div class="chat-bubble" style="background:{style['bg']}; border:1px solid {style['border']};">
                        <div class="chat-role">{style['label']}</div>
                        <div class="chat-msg">{raw_msg}</div>
                    </div>
                </div>
                """
            )

    def status_callback(session: SessionState):
        with status_lock:
            latest_status["session_id"] = session.session_id
            latest_status["patient_id"] = session.patient_id
            latest_status["current_stage"] = session.current_stage
            latest_status["current_agent"] = session.current_agent
            latest_status["exam_status"] = session.exam_status
            latest_status["last_error"] = session.last_error
            latest_status["sonographer_debug"] = session.sonographer_debug

    def on_user_input(text: str):
        session = SESSION_MANAGER.get()
        sid = session.session_id
        if not SESSION_MANAGER.is_valid(sid):
            return

        session.input_queue.put(text)
        session.conversation_history.append({
            "role": "user",
            "content": text
        })

    def submit(text):
        if text and text.strip():
            log_callback(text.strip(), agent="user", session_id=SESSION_MANAGER.get().session_id)
            on_user_input(text.strip())
        return ""

    def render_logs():
        with log_lock:
            html = "<div class='chat-container'>" + "".join(logs) + "</div>"
            html += '<div id="chat-end-anchor"></div>'
            return html

    def render_service_modules():
        with SERVICE_LOCK:
            orchestrator_info = SERVICE_STATUS.get("orchestrator", {"ok": False, "msg": "Unknown"})
            sonographer_info = SERVICE_STATUS.get("sonographer", {"ok": False, "msg": "Unknown"})
            radiologist_info = SERVICE_STATUS.get("radiologist", {"ok": False, "msg": "Unknown"})
            physician_info = SERVICE_STATUS.get("physician", {"ok": False, "msg": "Unknown"})

        def svc_row(label, info):
            badge_class = "svc-ok" if info["ok"] else "svc-bad"
            badge_text = "Online" if info["ok"] else "Offline"
            return f"""
            <div class="svc-row compact">
                <span class="svc-name">{label}</span>
                <span class="svc-badge {badge_class}">{badge_text}</span>
            </div>
            """

        return f"""
        <div class="status-box">
            <div class="status-label">Modules</div>
            <div class="status-value-small">
                {svc_row("Orchestrator", orchestrator_info)}
                {svc_row("Sonographer", sonographer_info)}
                {svc_row("Radiologist", radiologist_info)}
                {svc_row("Physician", physician_info)}
            </div>
        </div>
        """

    def render_basic_status():
        with status_lock:
            err_block = ""
            if latest_status["last_error"]:
                err_block = f"""
                <div class="status-box error-box">
                    <div class="status-label">Last Error</div>
                    <div class="status-value-small">{latest_status["last_error"]}</div>
                </div>
                """

        return f"""
        <div class="status-stack">
            <div class="status-inline-grid">
                <div class="status-box">
                    <div class="status-label">Session ID</div>
                    <div class="status-value">{latest_status["session_id"]}</div>
                </div>

                <div class="status-box">
                    <div class="status-label">Patient ID</div>
                    <div class="status-value">{latest_status["patient_id"]}</div>
                </div>
            </div>

            <div class="status-box">
                <div class="status-label">Current Agent</div>
                <div class="status-value">{latest_status["current_agent"]}</div>
            </div>

            {err_block}
        </div>
        """

    def render_debug_value():
        with status_lock:
            debug_on = latest_status.get("sonographer_debug", False)

        text = "Simulated" if debug_on else "Connected"
        cls = "debug-on" if debug_on else "debug-off"
        return f'<div class="debug-state-wrap"><div class="debug-state {cls}">{text}</div></div>'

    def render_debug_button_text():
        with status_lock:
            debug_on = latest_status.get("sonographer_debug", False)
        return "Connect" if debug_on else "Simulate"

    def refresh_services():
        initial_service_check()
        return render_service_modules()

    def toggle_sonographer_debug():
        session = SESSION_MANAGER.get()
        session.sonographer_debug = not session.sonographer_debug
        status_callback(session)
        return render_debug_value(), render_debug_button_text()

    def clean():
        new_session = SESSION_MANAGER.reset()
        new_session.sonographer_debug = False

        with log_lock:
            logs.clear()

        initial_service_check()
        status_callback(new_session)
        return "", render_basic_status(), render_debug_value(), render_debug_button_text(), render_service_modules()

    custom_css = """
    body {
        background: linear-gradient(180deg, #F6F8FC 0%, #EEF3F9 100%);
    }

    .app-shell {
        max-width: 1480px;
        margin: 0 auto;
        padding: 14px;
    }

    .hero-card {
        background: linear-gradient(135deg, #0F766E 0%, #2563EB 100%);
        border-radius: 24px;
        padding: 26px 30px;
        color: white;
        box-shadow: 0 14px 40px rgba(37, 99, 235, 0.20);
        margin-bottom: 16px;
    }

    .hero-title {
        font-size: 30px;
        font-weight: 800;
        margin-bottom: 8px;
        letter-spacing: -0.02em;
        color: #FFFFFF;
    }

    .hero-subtitle {
        font-size: 14px;
        opacity: 0.95;
        line-height: 1.7;
        color: #FFFFFF;
    }

    .panel-card {
        background: rgba(255, 255, 255, 0.88);
        backdrop-filter: blur(10px);
        border: 1px solid #E5E7EB;
        border-radius: 22px;
        box-shadow: 0 8px 30px rgba(15, 23, 42, 0.08);
        padding: 14px;
    }

    .panel-title {
        font-size: 16px;
        font-weight: 800;
        color: #111827;
        margin: 4px 4px 10px 4px;
    }

    .status-inline-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
    }

    #output_box {
        height: 680px;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        scroll-behavior: auto;
        box-sizing: border-box;
        padding: 14px;
        background: linear-gradient(180deg, #FBFCFE 0%, #F8FAFC 100%);
        border: 1px solid #EEF2F7;
        border-radius: 18px;
    }

    #output_box * {
        box-sizing: border-box;
    }

    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 12px;
        padding-bottom: 6px;
        max-width: 100%;
    }

    .chat-row {
        display: flex;
        width: 100%;
        min-width: 0;
    }

    .chat-row.left {
        justify-content: flex-start;
    }

    .chat-row.right {
        justify-content: flex-end;
    }

    .chat-bubble {
        max-width: 80%;
        border-radius: 18px;
        padding: 12px 14px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
        overflow: hidden;
    }

    .chat-role {
        font-size: 12px;
        font-weight: 800;
        color: #374151;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .chat-msg {
        font-size: 14px;
        line-height: 1.65;
        color: #111827;
        white-space: pre-wrap;
        word-break: break-word;
        overflow-wrap: anywhere;
        max-width: 100%;
    }

    .timeline-event {
        align-self: center;
        padding: 8px 14px;
        border-radius: 999px;
        background: #EEF2FF;
        border: 1px solid #CFD7FF;
        color: #334155;
        font-size: 12px;
        font-weight: 700;
        margin: 4px 0;
    }

    .status-stack {
        display: flex;
        flex-direction: column;
        gap: 12px;
    }

    .status-box {
        padding: 16px;
        border-radius: 18px;
        background: linear-gradient(180deg, #FFFFFF 0%, #F9FAFB 100%);
        border: 1px solid #E5E7EB;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.03);
    }

    .error-box {
        border: 1px solid #F3B3B3;
        background: linear-gradient(180deg, #FFF7F7 0%, #FFF1F1 100%);
    }

    .status-label {
        font-size: 12px;
        color: #6B7280;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 700;
    }

    .status-value {
        font-size: 20px;
        font-weight: 800;
        color: #111827;
        line-height: 1.3;
    }

    .status-value-small {
        font-size: 14px;
        font-weight: 600;
        color: #111827;
        line-height: 1.7;
        word-break: break-word;
    }

    .debug-row-wrap {
        border-radius: 18px;
        background: linear-gradient(180deg, #FFFFFF 0%, #F9FAFB 100%);
        border: 1px solid #E5E7EB;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.03);
        padding: 16px;
    }

    .debug-row {
        align-items: center !important;
        gap: 10px !important;
    }

    .debug-state-wrap {
        display: flex;
        align-items: center;
        min-height: 38px;
        overflow: hidden;
    }

    .debug-state {
        font-size: 18px;
        font-weight: 800;
        line-height: 1.2;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .debug-on {
        color: #B91C1C;
    }

    .debug-off {
        color: #166534;
    }

    .small-inline-btn button {
        width: 100% !important;
        height: 38px !important;
        border-radius: 10px !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        white-space: nowrap !important;
        padding: 0 10px !important;
    }

    .control-row {
        margin-top: 12px;
        align-items: end !important;
    }

    button {
        border-radius: 14px !important;
        font-weight: 700 !important;
        height: 48px !important;
    }

    textarea {
        border-radius: 16px !important;
        font-size: 14px !important;
    }

    .example-short-btn button {
        height: 42px !important;
        border-radius: 12px !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        white-space: nowrap !important;
    }

    .svc-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        padding: 8px 0;
        border-bottom: 1px solid #EEF2F7;
    }

    .svc-row:last-child {
        border-bottom: none;
        padding-bottom: 0;
    }

    .svc-name {
        font-size: 14px;
        font-weight: 700;
        color: #111827;
    }

    .svc-badge {
        font-size: 11px;
        font-weight: 800;
        padding: 4px 8px;
        border-radius: 999px;
        line-height: 1;
        white-space: nowrap;
    }

    .svc-ok {
        background: #EAF8EF;
        color: #166534;
        border: 1px solid #B7E4C7;
    }

    .svc-bad {
        background: #FEF2F2;
        color: #B91C1C;
        border: 1px solid #F5C2C7;
    }
    """

    head_js = """
    <script>
    function setupChatAutoScroll() {
        const wrapper = document.getElementById("output_box");
        if (!wrapper) return;

        const BOTTOM_THRESHOLD = 150;

        if (wrapper.dataset.bound === "1") return;
        wrapper.dataset.bound = "1";

        let autoScroll = true;

        function distanceFromBottom() {
            return wrapper.scrollHeight - wrapper.scrollTop - wrapper.clientHeight;
        }

        function scrollToBottom(force = false) {
            if (force || autoScroll) {
                requestAnimationFrame(() => {
                    wrapper.scrollTop = wrapper.scrollHeight;
                });
            }
        }

        wrapper.addEventListener("scroll", () => {
            autoScroll = distanceFromBottom() <= BOTTOM_THRESHOLD;
        });

        const observer = new MutationObserver(() => {
            if (autoScroll) {
                scrollToBottom(true);
            }
        });

        observer.observe(wrapper, {
            childList: true,
            subtree: true,
            characterData: true
        });

        scrollToBottom(true);
    }

    setInterval(setupChatAutoScroll, 500);
    document.addEventListener("DOMContentLoaded", setupChatAutoScroll);
    window.addEventListener("load", setupChatAutoScroll);
    </script>
    """

    with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), head=head_js) as demo:
        with gr.Column(elem_classes="app-shell"):
            gr.HTML("""
                <div class="hero-card">
                    <div class="hero-title">🤖 IS-MAN: Multi-Agent Ultrasound Assistant</div>
                    <div class="hero-subtitle">
                        A clinical workflow console for autonomous sonography.
                    </div>
                </div>
            """)

            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Column(elem_classes="panel-card"):
                        gr.HTML('<div class="panel-title">Conversation</div>')

                        output_box = gr.HTML(
                            value=render_logs(),
                            elem_id="output_box"
                        )

                        with gr.Row(elem_classes="control-row", equal_height=True):
                            input_box = gr.Textbox(
                                placeholder="Please enter your question...",
                                show_label=False,
                                lines=3,
                                scale=7,
                                container=False
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1)
                            clean_btn = gr.Button("Reset", scale=1)

                with gr.Column(scale=1):
                    with gr.Column(elem_classes="panel-card"):
                        gr.HTML('<div class="panel-title">Session Status</div>')

                        basic_status_box = gr.HTML(value=render_basic_status())

                        with gr.Column(elem_classes="debug-row-wrap"):
                            gr.HTML('<div class="status-label">Robot Execution</div>')
                            with gr.Row(elem_classes="debug-row", equal_height=True):
                                with gr.Column(scale=3, min_width=140):
                                    debug_value_box = gr.HTML(value=render_debug_value())
                                with gr.Column(scale=2, min_width=110):
                                    toggle_debug_btn = gr.Button(
                                        value=render_debug_button_text(),
                                        variant="secondary",
                                        elem_classes="small-inline-btn"
                                    )

                        service_box = gr.HTML(value=render_service_modules())
                        refresh_services_btn = gr.Button("Refresh Services", variant="secondary")

                    with gr.Column(elem_classes="panel-card"):
                        gr.HTML('<div class="panel-title">Example Input</div>')
                        ex_btn_1 = gr.Button(
                            "Example 1",
                            variant="secondary",
                            elem_classes="example-short-btn"
                        )

        send_btn.click(fn=submit, inputs=input_box, outputs=input_box)
        input_box.submit(fn=submit, inputs=input_box, outputs=input_box)

        clean_btn.click(
            fn=clean,
            inputs=[],
            outputs=[input_box, basic_status_box, debug_value_box, toggle_debug_btn, service_box]
        )

        toggle_debug_btn.click(
            fn=toggle_sonographer_debug,
            inputs=[],
            outputs=[debug_value_box, toggle_debug_btn]
        )

        refresh_services_btn.click(
            fn=refresh_services,
            inputs=[],
            outputs=service_box
        )

        ex_btn_1.click(
            fn=lambda: "I want to perform a thyroid ultrasound examination. I don't have thyroid surgery before.",
            inputs=[],
            outputs=input_box
        )

        timer = gr.Timer(0.5)
        timer.tick(fn=render_logs, inputs=None, outputs=output_box)
        timer.tick(fn=render_basic_status, inputs=None, outputs=basic_status_box)
        timer.tick(fn=render_debug_value, inputs=None, outputs=debug_value_box)
        timer.tick(fn=render_debug_button_text, inputs=None, outputs=toggle_debug_btn)
        timer.tick(fn=render_service_modules, inputs=None, outputs=service_box)

    initial_service_check()

    threading.Thread(
        target=agent_main_loop,
        args=(log_callback, status_callback),
        daemon=True
    ).start()

    demo.launch()


if __name__ == "__main__":
    launch_gradio()