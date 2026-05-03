from langchain_core.prompts import ChatPromptTemplate
from prompt.physician_prompt import system_prompt
from utils import RAGRetriever
from langchain_core.output_parsers import StrOutputParser
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
        self.patient_ID = str(patient_ID)

        folder = "./patient"
        self.report_dir = Path(folder) / self.patient_ID / "report.txt"

        self.logger = logger or (lambda *a, **k: None)

        self.rag_retriever = RAGRetriever(
            db_path=rag_db_path,
            embedding_model=embedding_model
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "[Retrieved Context]\n{context}\n\n[Ultrasound Report]\n{report}")
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def _load_ultrasound_report(self) -> str | None:
        if not self.report_dir.exists():
            return None
        return self.report_dir.read_text(encoding="utf-8").strip()

    def _ask_user_for_report(self, user_input_queue=None) -> str:
        if user_input_queue is None:
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            return "\n".join(lines).strip()

        self.logger("Please enter your report...", agent="system")
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
            "report": report,
        })

        self.logger(output, agent="physician")
        return {
            "agent": self.name,
            "content": output
        }