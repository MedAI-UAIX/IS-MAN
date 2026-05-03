from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from utils import extract_json
from prompt.orchestrator_prompt import system_prompt


class OrchestratorAgent:
    def __init__(self, llm: ChatOpenAI):
        self.name = "orchestrator"
        self.llm = llm
        self.allowed_agents = ["sonographer", "radiologist", "physician"]
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{full_input}")
        ])

    def _format_history(self, conversation_history):
        if not conversation_history:
            return ""

        lines = []
        for item in conversation_history:
            role = item.get("role", "unknown")
            content = item.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def run(self, user_input: str, conversation_history=None):
        history_text = self._format_history(conversation_history)

        if history_text:
            full_input = (
                f"[Conversation History]\n{history_text}\n\n"
                f"[Current Input]\n{user_input}"
            )
        else:
            full_input = user_input

        messages = self.prompt.format_messages(full_input=full_input)
        response = self.llm.invoke(messages).content
        parsed = extract_json(response)

        if not parsed:
            return {
                "content": response,
                "tool_calls": {"action": None, "agent": None, "input": None}
            }
        return parsed