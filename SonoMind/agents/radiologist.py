from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from utils import extract_json,user_input_queue
from prompt.radiologist_prompt import *
import json
import re
from tools.radiologist_tools import *
import os
import base64
import ast

class RadiologistAgent:
    def __init__(self, llm, logger=None, patient_ID="001"):
        self.name = "radiologist"
        self.llm = llm

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{full_input}")
        ])
        self.patient_ID=patient_ID
        self.follow_up_tool = FollowUpTool(patient_ID=self.patient_ID)
        self.MMOE_tool = MMOETool(patient_ID=self.patient_ID)
        self.thynet_tool = ThynetModel(patient_ID=self.patient_ID)
        self.thynetS_tool = ThynetSModel(patient_ID=self.patient_ID)
        self.logger = logger or (lambda *a, **k: None)
    
    def log(self, msg):
        self.logger(msg, agent="radiologist")

    def sys(self, msg):
        self.logger(msg, agent="system")

    def user(self, msg):
        self.logger(msg, agent="user")

    def ask_user(self, question, user_input_queue):
        self.sys(question)
        return user_input_queue.get()   

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

        self.sys(f"[Radiologist calling tool] {tool_name} args={arguments}")

        if tool_name == "ImageAnalysis":
            model = arguments.get("model")
            if model == "thynet":
                result = self.thynet_tool.predict()["summary_sentence"]
                self.sys(f"[Thynet model] {result}")
                return result
            
            if model == "thynet-s":
                result = self.thynetS_tool.predict()["summary_sentence"]
                self.sys(f"[Thynet-S model] {result}")
                return result

            if model == "MMOE":
                result = self.MMOE_tool.predict()["summary_sentence"]
                return result

            if model == "follow-up LLM":
                prev_report = self.ask_user(
                    "[Radiologist]: Please provide the previous thyroid ultrasound report.",
                    user_input_queue
                )             
                surgery_history = self.ask_user(
                    "[Radiologist]: Do you have any history of thyroid surgery?",
                    user_input_queue
                )
                result = self.follow_up_tool.run_follow_up(        
                    prev_report=prev_report,
                    surgery_history=surgery_history)
                self.sys(f"[follow-up model] {result}")
                return result

        if tool_name == "transfer_to_supervisor":
            summary = arguments.get("input")
            return {
                "state": "success",
                "tool": f"transfer_to_supervisor(input={summary})"
            }

        return {"state": "fail", "tool": f"{tool_name} unknown"}

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def find_images(self, folder=r"SonoMind/patient"):
        exts = [".png", ".jpg", ".jpeg", ".bmp"]
        img_folder = os.path.join(folder,self.patient_ID)
        return [
            os.path.join(img_folder, f)
            for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in exts
        ]

    def run_once(self, context):
        analysis_text = ""
        if context.get("analysis_result"):
            analysis_text = (
                "Analysis results:\n"
                f"{context['analysis_result']}\n\n"
            )

        full_input = (
            f"Request:\n{context.get('user_input', '')}\n\n"
            + analysis_text
        )

        messages = self.prompt.format_messages(full_input=full_input)
        user_msg = messages[1]
        text_content = user_msg.content
        content_list = [
    {"type": "text", "text": text_content}
]
        images = context.get("images", [])
        for img_path in images:
            base64_image = self.encode_image(img_path)
            content_list.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
    })
        messages[1] = {
            "role": "user",
            "content": content_list
        }


        return self.llm.invoke(messages).content

    def loop(self, user_input: str):
        images = self.find_images()

        context = {
            "user_input": user_input,
            "images": images,
            "analysis results": []
        }

        while True:
            raw_output = self.run_once(context)
            parsed = self.safe_json_parse(raw_output)
            self.log(f"[Radiologist]: {parsed}")
            tool_call = parsed.get("tool_calls", {})

            if tool_call and tool_call["name"] != "transfer_to_supervisor":
                result = self.call_tool(tool_call["name"], tool_call["arguments"])

                context["analysis results"].append(result)
                continue

            if tool_call and tool_call["name"] == "transfer_to_supervisor":
                summary = tool_call["arguments"]["input"]
                folder = r"SonoMind/patient"
                report_dir = os.path.join(folder,self.patient_ID,"report.txt")
                with open(report_dir, "w",encoding="utf-8") as f:
                    f.write(summary)
                return {
                    "agent": "radiologist",
                    "input": summary
                }

            
