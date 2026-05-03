import json
import re
import os
import ast
import base64
from langchain_core.prompts import ChatPromptTemplate

from prompt.radiologist_prompt import system_prompt
from tools.radiologist_tools import FollowUpTool, MMOETool, ThynetModel, ThynetSModel


class RadiologistAgent:
    def __init__(self, llm, logger=None, patient_ID="001"):
        self.name = "radiologist"
        self.llm = llm
        self.logger = logger or (lambda *a, **k: None)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{full_input}")
        ])

        self.patient_ID = str(patient_ID)
        self._build_tools()

    def _build_tools(self):
        self.follow_up_tool = FollowUpTool(patient_ID=self.patient_ID)
        self.MMOE_tool = MMOETool(patient_ID=self.patient_ID)
        self.thynet_tool = ThynetModel(patient_ID=self.patient_ID)
        self.thynetS_tool = ThynetSModel(patient_ID=self.patient_ID)

    def set_patient_id(self, patient_id):
        self.patient_ID = str(patient_id)
        self._build_tools()

    def log(self, msg):
        self.logger(msg, agent="radiologist")

    def sys(self, msg):
        self.logger(msg, agent="system")

    def user(self, msg):
        self.logger(msg, agent="user")

    def _append_history(self, context, role, content):
        context.setdefault("conversation_history", []).append({
            "role": role,
            "content": str(content)
        })

    def _format_history(self, context):
        history = context.get("conversation_history", [])
        if not history:
            return ""

        lines = []
        for item in history:
            role = item.get("role", "unknown")
            content = item.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def ask_user(self, question, user_input_queue, context):
        self.sys(question)
        self._append_history(context, "radiologist", question)

        if user_input_queue is None:
            raise RuntimeError("RadiologistAgent.ask_user requires user_input_queue, but got None.")

        user_reply = user_input_queue.get()
        self.user(user_reply)
        self._append_history(context, "user", user_reply)
        return user_reply

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
            print("JSON error, oriniginal data:")
            print(data)
            raise

    def call_tool(self, tool_name, arguments, context, user_input_queue=None):
        self.sys(f"[Radiologist calling tool] {tool_name} args={arguments}")
        self._append_history(context, "tool_call", f"{tool_name} args={arguments}")

        if tool_name == "ImageAnalysis":
            model = arguments.get("model")

            if model == "thynet":
                result = self.thynet_tool.predict()["summary_sentence"]
                self.sys(f"[Thynet model] {result}")
                self._append_history(context, "tool_result", f"thynet: {result}")
                return result

            if model == "thynet-s":
                result = self.thynetS_tool.predict()["summary_sentence"]
                self.sys(f"[Thynet-S model] {result}")
                self._append_history(context, "tool_result", f"thynet-s: {result}")
                return result

            if model == "MMOE":
                result = self.MMOE_tool.predict()["summary_sentence"]
                self.sys(f"[Nodule features from the MMOE model] {result}")
                self._append_history(context, "tool_result", f"MMOE: {result}")
                return result

            if model == "follow-up LLM":
                prev_report = self.ask_user(
                    "Please provide the previous thyroid ultrasound report.",
                    user_input_queue,
                    context
                )
                surgery_history = self.ask_user(
                    "Do you have any history of thyroid surgery?",
                    user_input_queue,
                    context
                )

                result = self.follow_up_tool.run_follow_up(
                    prev_report=prev_report,
                    surgery_history=surgery_history
                )
                self.sys(f"[follow-up model] {result}")
                self._append_history(context, "tool_result", f"follow-up LLM: {result}")
                return result

        if tool_name == "transfer_to_supervisor":
            summary = arguments.get("input")
            self._append_history(context, "tool_result", f"transfer_to_supervisor: {summary}")
            return {
                "state": "success",
                "tool": f"transfer_to_supervisor(input={summary})"
            }

        fail_msg = {"state": "fail", "tool": f"{tool_name} unknown"}
        self._append_history(context, "tool_result", str(fail_msg))
        return fail_msg

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def find_images(self, folder="./patient"):
        exts = [".png", ".jpg", ".jpeg", ".bmp"]
        img_folder = os.path.join(folder, self.patient_ID)

        if not os.path.isdir(img_folder):
            self.sys(f"Image folder not found: {img_folder}")
            return []

        return [
            os.path.join(img_folder, f)
            for f in os.listdir(img_folder)
            if os.path.splitext(f)[1].lower() in exts
        ]

    def run_once(self, context):
        analysis_text = ""
        if context.get("analysis_results"):
            analysis_text = (
                "Analysis results:\n"
                f"{context['analysis_results']}\n\n"
            )

        history_text = self._format_history(context)
        if history_text:
            history_text = f"Conversation history:\n{history_text}\n\n"

        full_input = (
            f"Request:\n{context.get('user_input', '')}\n\n"
            + history_text
            + analysis_text
        )
        print("full_input: ", full_input)
        messages = self.prompt.format_messages(full_input=full_input)
        user_msg = messages[1]
        text_content = user_msg.content

        content_list = [{"type": "text", "text": text_content}]

        images = context.get("images", [])
        for img_path in images:
            try:
                base64_image = self.encode_image(img_path)
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
            except Exception as e:
                self.sys(f"Failed to encode image {img_path}: {e}")

        messages[1] = {
            "role": "user",
            "content": content_list
        }

        return self.llm.invoke(messages).content

    def loop(self, user_input: str, user_input_queue=None):
        images = self.find_images()

        context = {
            "user_input": user_input,
            "images": images,
            "analysis_results": [],
            "conversation_history": [],
            "mmoe_called": False
        }

        self._append_history(context, "user", user_input)

        while True:
            raw_output = self.run_once(context)
            parsed = self.safe_json_parse(raw_output)
            self.log(parsed)

            content = parsed.get("content", "")
            if content:
                self._append_history(context, "radiologist", content)

            tool_call = parsed.get("tool_calls", {})

            if not tool_call:
                self.sys("No tool_calls found, retrying...")
                continue

            tool_name = tool_call.get("name")
            arguments = tool_call.get("arguments", {})

            if (
                tool_name == "ImageAnalysis"
                and arguments.get("model") == "MMOE"
                and context["mmoe_called"]
            ):
                self.sys("MMOE has already been called once. Preventing repeated loop.")
                summary = content or "Radiology analysis completed."
                return {
                    "agent": "radiologist",
                    "input": summary
                }

            if tool_name != "transfer_to_supervisor":
                result = self.call_tool(
                    tool_name,
                    arguments,
                    context,
                    user_input_queue=user_input_queue
                )

                if tool_name == "ImageAnalysis" and arguments.get("model") == "MMOE":
                    context["mmoe_called"] = True

                context["analysis_results"].append(result)
                continue

            summary = arguments.get("input", "")
            report_content = content or summary

            folder = "./patient"
            patient_dir = os.path.join(folder, self.patient_ID)
            os.makedirs(patient_dir, exist_ok=True)

            report_dir = os.path.join(patient_dir, "report.txt")
            with open(report_dir, "w", encoding="utf-8") as f:
                f.write(report_content)

            return {
                "agent": "radiologist",
                "input": summary
            }