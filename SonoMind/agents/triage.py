from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from utils import extract_json
from prompt.triage_prompt import *

class TriageAgent:
    def __init__(self, llm: ChatOpenAI):
        self.name = "triage"
        self.llm = llm
        self.allowed_agents = ["sonographer", "radiologist", "physician"]
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{user_input}")
        ])

    def run(self, user_input: str):
        messages = self.prompt.format_messages(user_input=user_input)
        response = self.llm.invoke(messages).content
        parsed = extract_json(response)
        if not parsed:
            return {
                "content": response,
                "tool_calls": {"action": None, "agent": None, "input": None}
            }
        return parsed
