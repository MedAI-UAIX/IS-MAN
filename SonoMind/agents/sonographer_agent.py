import os
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from SonoMind.prompts.triage_prompt import *
from SonoMind.tools import *

llm = ChatOpenAI(
    api_key="0",
    base_url="http://localhost:{}/v1".format(os.environ.get("API_PORT", 8001)),
    model="sonographer_agent"
)

system = system_prompt


prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "{user_input}")
])

agent = initialize_agent(
    tools,
    llm,
    verbose=True
)

user_input = "..."
response = agent.run(user_input)
print(response)