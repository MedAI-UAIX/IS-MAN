import json
import re
import ast
from queue import Empty
from langchain_core.prompts import ChatPromptTemplate
from prompt.sonographer_prompt import system_prompt
from tools.FrankaRequest import FrankaClient, FlaskServerState


class SonographerAgent:
    def __init__(self, llm, logger=None, debug_mode=False):
        self.name = "sonographer"
        self.llm = llm
        self.debug_mode = debug_mode
        self.max_parse_attempts = 3

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{full_input}")
        ])

        self.logger = logger or (lambda *a, **k: None)

        self.frankaclient = None
        self.frankaState = None
        if not self.debug_mode:
            self.frankaclient = FrankaClient(port=5004)
            self.frankaState = FlaskServerState(port=5006)
            self.frankaState.start()

    def set_debug_mode(self, debug_mode: bool):
        if self.debug_mode == debug_mode:
            return

        self.debug_mode = debug_mode

        if self.debug_mode:
            self.frankaclient = None
            self.frankaState = None
            self.sys("[Sonographer] Switched to DEBUG mode.")
        else:
            self.frankaclient = FrankaClient(port=5004)
            self.frankaState = FlaskServerState(port=5006)
            self.frankaState.start()
            self.sys("[Sonographer] Switched to REAL mode.")

    def log(self, msg):
        self.logger(msg, agent="sonographer")

    def sys(self, msg):
        self.logger(msg, agent="system")

    def user(self, msg):
        self.logger(msg, agent="user")

    def run_once(self, agent_input):
        full_input = (
            f"Patient information:\n{agent_input.get('patient_info', '')}\n\n"
            f"Execution history:\n{agent_input.get('execution_history', [])}\n\n"
            f"Nodule detection:\n{agent_input.get('nodule_detection', [])}\n\n"
            f"User input:\n{agent_input.get('user_input', '')}"
        )
        messages = self.prompt.format_messages(full_input=full_input)
        return self.llm.invoke(messages).content

    def safe_json_parse(self, data):
        if isinstance(data, dict):
            return data

        if not isinstance(data, str):
            return {
                "error": f"Expect dict or str, got {type(data)}",
                "raw": str(data)
            }

        cleaned = data.strip()
        cleaned = re.sub(r"```json|```python|```", "", cleaned).strip()
        cleaned = cleaned.replace('"""', "").strip()

        if not cleaned:
            return {
                "error": "Empty model output after cleaning",
                "raw": data
            }

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        try:
            return ast.literal_eval(cleaned)
        except Exception as e:
            print("JSON error, oriniginal data:")
            print(repr(data))
            print("cleaned repr =", repr(cleaned))
            return {
                "error": f"JSON/dict parse failed: {type(e).__name__}: {e}",
                "raw": data
            }

    def _mock_tool_result(self, tool_desc: str):
        return {"state": "success", "tool": tool_desc}

    def _real_tool_result(self, api_call, tool_desc: str):
        api_call()
        franka_state = self.frankaState.wait_for_state()
        print("frankastate", franka_state)
        return {"state": franka_state, "tool": tool_desc}

    def _execute_robot_tool(self, api_call, tool_desc: str):
        if self.debug_mode:
            return self._mock_tool_result(tool_desc)
        return self._real_tool_result(api_call, tool_desc)

    def call_tool(self, tool_name, arguments, context):
        if tool_name == "ControlRobotArm":
            action = arguments.get("action")
            tool_desc = f"ControlRobotArm(action={action})"
            return self._execute_robot_tool(
                api_call=lambda: self.frankaclient.ControlRobotArm(action=action),
                tool_desc=tool_desc
            )

        if tool_name == "ControlForce":
            action = arguments.get("action")
            tool_desc = f"ControlForce(action={action})"
            return self._execute_robot_tool(
                api_call=lambda: self.frankaclient.ControlForce(action=action),
                tool_desc=tool_desc
            )

        if tool_name == "ControlMotion":
            action = arguments.get("action")
            tool_desc = f"ControlMotion(action={action})"
            return self._execute_robot_tool(
                api_call=lambda: self.frankaclient.ControlMotion(action=action),
                tool_desc=tool_desc
            )

        if tool_name == "SwitchControlMode":
            mode = arguments.get("mode")
            tool_desc = f"SwitchControlMode(mode={mode})"
            return self._execute_robot_tool(
                api_call=lambda: self.frankaclient.SwitchControlMode(mode=mode),
                tool_desc=tool_desc
            )

        if tool_name == "DetectKeypoints":
            region = arguments.get("region")
            tool_desc = f"DetectKeypoints(region={region})"
            return self._execute_robot_tool(
                api_call=lambda: self.frankaclient.DetectKeypoints(region=region),
                tool_desc=tool_desc
            )

        if tool_name == "UltrasoundScan":
            organ = arguments.get("organ")
            region = arguments.get("region")
            direction = arguments.get("direction")

            if organ == "thyroid" and direction == "longitudinal":
                context.setdefault("nodule_detection", []).append({
                    "organ": organ,
                    "region": region,
                    "direction": direction
                })

            tool_desc = (
                f"UltrasoundScan(organ={organ}, region={region}, direction={direction})"
            )
            return self._execute_robot_tool(
                api_call=lambda: self.frankaclient.UltrasoundScan(
                    organ=organ,
                    region=region,
                    direction=direction
                ),
                tool_desc=tool_desc
            )

        if tool_name == "transfer_to_supervisor":
            summary = arguments.get("input")
            return {"state": "success", "tool": f"transfer_to_supervisor(input={summary})"}

        return {"state": "fail", "tool": f"{tool_name} unknown"}

    def loop(self, initial_input, user_input_queue=None):
        context = initial_input

        context.setdefault("patient_info", "")
        context.setdefault("execution_history", [])
        context.setdefault("nodule_detection", [])
        context.setdefault("user_input", "")

        while True:
            try:
                if user_input_queue is not None:
                    user_msg = user_input_queue.get_nowait()
                    context["user_input"] = user_msg
                    self.user(f"[User received]: {user_msg}")
                else:
                    context["user_input"] = ""
            except Empty:
                context["user_input"] = ""

            parsed = None
            last_error = None
            last_raw_output = None

            for attempt in range(1, self.max_parse_attempts + 1):
                raw_output = self.run_once(context)
                last_raw_output = raw_output
                self.log(raw_output)

                parsed = self.safe_json_parse(raw_output)

                if not parsed.get("error"):
                    break

                last_error = parsed.get("error")
                self.sys(
                    f"[Parse Error] attempt {attempt}/{self.max_parse_attempts}: {last_error}"
                )

            if parsed is None or parsed.get("error"):
                raise RuntimeError(
                    f"Sonographer model output parse failed after {self.max_parse_attempts} attempts. "
                    f"Last error: {last_error}. "
                    f"Last raw output: {repr(last_raw_output)}"
                )

            tool_call = parsed.get("tool_calls")

            if not tool_call:
                raise RuntimeError(
                    f"Sonographer output has no tool_calls after successful parse. "
                    f"Parsed output: {parsed}"
                )

            tool_name = tool_call.get("name")
            arguments = tool_call.get("arguments", {})

            if tool_name != "transfer_to_supervisor":
                self.sys(f"[Tool calling] {tool_call}")
                result = self.call_tool(tool_name, arguments, context)
                context["execution_history"].append(result)
                context["user_input"] = ""
                self.sys(f"[Sonographer context]: {context}")
                continue

            if not context["nodule_detection"]:
                scan_result_summary = "No abnormalities detected"
            else:
                scan_result_summary = context["nodule_detection"]

            return_input = arguments.get("input", "")
            transfer_text = {
                "agent": "sonographer",
                "input": return_input,
                "scan_result_summary": scan_result_summary
            }
            return transfer_text