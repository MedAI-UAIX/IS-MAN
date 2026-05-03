import json
import queue
from sentence_transformers import SentenceTransformer
import chromadb

# ==== 全局对象 ====
user_input_queue = queue.Queue()
# interrupt_flag = False


def extract_json(text: str):
    """从文本中提取 JSON"""
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except:
        return None

def normalize_agent(agent_raw: str, allowed_agents=None):
    """将 agent 名称归一化，例如 'Physician Agent' -> 'physician'"""
    if allowed_agents is None:
        allowed_agents = ["sonographer", "radiologist", "physician"]
    t = agent_raw.lower()
    for ag in allowed_agents:
        if ag in t:
            return ag
    return None

class RAGRetriever:
    def __init__(
        self,
        db_path: str,
        embedding_model: str,
        collection_name: str = "thyroid_rules_structured"
    ):
        self.encoder = SentenceTransformer(embedding_model)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(collection_name)

    def retrieve(self, query: str, top_k: int = 3):
        query_embedding = self.encoder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        return results["documents"][0]
