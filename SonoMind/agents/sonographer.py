import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from prompt.sonographer_prompt import *
import re
import ast
from utils import user_input_queue
from queue import Empty
from tools.sonographer_tools import *
    
class SonographerAgent:
    def __init__(self,tools,llm, logger=None):
        self.name = "sonographer"
        self.llm = llm
        self.tools = tools
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{full_input}")
        ])
        self.logger = logger or (lambda *a, **k: None)
 
    def log(self, msg):
        self.logger(msg, agent="sonographer")

    def sys(self, msg):
        self.logger(msg, agent="system")

    def user(self, msg):
        self.logger(msg, agent="user")

    def run_once(self, agent_input):
        full_input = (
            f"Patient information:\n{agent_input.get('patient_info','')}\n\n"
            f"Execution history:\n{agent_input.get('execution_history',[])}\n\n"
            f"Nodule detection:\n{agent_input.get('nodule_detection',[])}\n\n"
            f"User input:\n{agent_input.get('user_input','')}"
        )

        messages = self.prompt.format_messages(full_input=full_input)
        return self.llm.invoke(messages).content 
    
    def safe_json_parse(self, data):
        if isinstance(data, dict):
            return data

        if not isinstance(data, str):
            raise TypeError(f"Expect dict or str, got {type(data)}")

        cleaned = data.strip()
        cleaned = re.sub(r"```json|```", "", cleaned).strip()
        cleaned = cleaned.replace('"""', "").strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        try:
            return ast.literal_eval(cleaned)
        except Exception:
            print(data)
            raise

    def call_tool(self, tool_name, arguments):
        if tool_name == "ControlRobotArm":
            res = ControlRobotArm(**arguments) 
            return {"state": res["status"], "tool": f"ControlRobotArm({arguments})", "message": res["message"]}

        if tool_name == "ControlForce":
            res = ControlForce(**arguments)
            return {"state": res["status"], "tool": f"ControlForce({arguments})", "message": res["message"]}

        if tool_name == "ControlMotion":
            res = ControlMotion(**arguments)
            return {"state": res["status"], "tool": f"ControlMotion({arguments})", "message": res["message"]}

        if tool_name == "SwitchControlMode":
            res = SwitchControlMode(**arguments)
            return {"state": res["status"], "tool": f"SwitchControlMode({arguments})", "message": res["message"]}

        if tool_name == "DetectKeypoints":
            res = DetectKeypoints(**arguments)
            return {"state": res["status"], "tool": f"DetectKeypoints({arguments})", "message": res["message"]}


        if tool_name == "UltrasoundScan":
            organ = arguments.get("organ")
            region = arguments.get("region")
            direction = arguments.get("direction")
            try:
                res = UltrasoundScan(
                    organ=organ,
                    region=region,
                    direction=direction,
                    agent=self
                )
                state = res["status"]
                message = res.get("message", "")
            except Exception as e:
                state = "error"
                message = str(e)

            return {
                "state": state,
                "tool": f"UltrasoundScan(organ={organ}, region={region}, direction={direction})",
                "message": message
            }

        if tool_name == "transfer_to_supervisor":
            summary = arguments.get("input")
            return {"state": "success", "tool": f"transfer_to_supervisor(input={summary})"}

        return {"state": "fail", "tool": f"{tool_name} unknown"}

    def loop(self, initial_input):
        context = initial_input
        while True:
            try:
                user_msg = user_input_queue.get_nowait()
                context["user_input"] = user_msg
                self.user(f"[User received]: {user_msg}")
            except Empty:
                context["user_input"] = ""

            raw_output = self.run_once(context)
            self.log(f"[Sonographer]: {raw_output}")
            parsed = self.safe_json_parse(raw_output)

            tool_call = parsed.get("tool_calls")

            if tool_call and tool_call["name"] != "transfer_to_supervisor":
                self.sys(f"[Sonographer calling tool] {tool_call}")
                result = self.call_tool(tool_call["name"], tool_call["arguments"])
                context["execution_history"].append(result)
                context["user_input"] = ""
                self.sys(f"[Sonographer]: {context}")
                continue

            if tool_call and tool_call["name"] == "transfer_to_supervisor":
                if not context["nodule_detection"]:
                    scan_result_summary = "No abnormalities detected"
                elif context["nodule_detection"]:
                    scan_result_summary = context["nodule_detection"]
                return_input = tool_call["arguments"]["input"]
                transfer_text = {'agent': 'sonographer', 'input': return_input, 'scan_result_summary': scan_result_summary}
                return transfer_text