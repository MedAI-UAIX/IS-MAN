from langchain_core.prompts import ChatPromptTemplate
from prompt.physician_prompt import *
from utils import RAGRetriever
from langchain_core.output_parsers import StrOutputParser
import os
from pathlib import Path

class PhysicianAgent:
    def __init__(
        self,
        llm,
        rag_db_path: str,
        embedding_model: str,
        patient_ID="001",
        logger=None
    ):
        self.name = "physician"
        self.llm = llm
        self.patient_ID=patient_ID
        folder = r"SonoMind/patient"
        report_dir = os.path.join(folder,self.patient_ID,"report.txt")
        self.report_dir = report_dir
        self.logger = logger or (lambda *a, **k: None)
        self.rag_retriever = RAGRetriever(
            db_path=rag_db_path,
            embedding_model=embedding_model
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "[Ultrasound Report]\n{report}")
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def _load_ultrasound_report(self) -> str | None:
        if not self.report_dir.exists():
            return None

        txt_files = list(self.report_dir.glob("*.txt"))
        if not txt_files:
            return None

        return txt_files[0].read_text(encoding="utf-8")


    def _ask_user_for_report(self, user_input_queue=None) -> str:
        if user_input_queue is None:
            lines = []
            while True:
                line = input()
                if not line:
                    break  
                lines.append(line)
            return "\n".join(lines).strip()

        self.logger("waiting for report...")
        report = user_input_queue.get()
        return report.strip()


    def run(self, user_input_queue=None):
        report = self._load_ultrasound_report()

        if not report:
            report = self._ask_user_for_report(user_input_queue=user_input_queue)

        docs = self.rag_retriever.retrieve(report)
        context = "\n\n".join(docs)

        output = self.chain.invoke({
            "context": context,
            "report": report
        })

        self.logger(output, agent="physician")
        return {
            "agent": self.name,
            "content": output
        }