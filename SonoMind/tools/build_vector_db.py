from sentence_transformers import SentenceTransformer
import chromadb
import uuid
import json

# ========================
# 1. 配置路径
# ========================
RULES_JSON = r"/home/usai/auto_RUSS/real_implement/tools/database/thyroid_rules.json"
CHROMA_PATH = r"/home/usai/auto_RUSS/real_implement/tools/vector_db_docx_pubmedbert-base_structured/chroma"
MODEL_NAME = r"/home/usai/auto_RUSS/real_implement/tools/pubmedbert-base"

# ========================
# 2. 加载结构化规则
# ========================
print("🔍 正在加载结构化规则...")
with open(RULES_JSON, 'r', encoding='utf-8') as f:
    rules = json.load(f)

print(f"✅ 加载 {len(rules)} 条结构化规则")

# ========================
# 3. 初始化 embedding 模型
# ========================
print(f"🧠 正在加载 embedding 模型 ({MODEL_NAME})...")
embedding_model = SentenceTransformer(MODEL_NAME)

# ========================
# 4. 构建 Chroma 向量数据库
# ========================
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection("thyroid_rules_structured")

documents = []
metadatas = []
ids = []

for rule in rules:
    # 拼接成完整文本用于 embedding
    doc_text = f"TI-RADS {rule['tirds_level']} ({rule['points']}): {rule['recommendation']}"

    metadata = {
        "tirds_level": int(rule["tirds_level"]) if rule["tirds_level"] is not None else "unknown",
        "section": str(rule["section"]) if rule["section"] is not None else "unknown",
        "action": str(rule["action"]) if rule["action"] is not None else "unknown",
        "size_threshold_cm": float(rule["size_threshold_cm"]) if rule["size_threshold_cm"] is not None else 99,
    }

    documents.append(doc_text)
    metadatas.append(metadata)
    ids.append(str(uuid.uuid4()))

# 生成 embeddings
embeddings = embedding_model.encode(documents).tolist()

# 添加到数据库
collection.add(
    embeddings=embeddings,
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print(f"""
✅ 结构化向量数据库构建成功！
📁 路径: {CHROMA_PATH}
📦 集合: thyroid_rules_structured
🔢 规则数量: {len(rules)}
🚀 下一步: 创建 RAG 检索器
""")