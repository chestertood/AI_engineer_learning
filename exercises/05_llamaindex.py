"""4.2 LlamaIndex: RAG แบบเดียวกับ 04_ex3.py (attention.pdf + MiniLM + Claude)

ต้องมี ANTHROPIC_API_KEY ใน .env
"""

from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic

load_dotenv(Path(__file__).parent.parent / ".env")

PDF = Path(__file__).parent / "data" / "attention.pdf"

# Settings = ค่า global (LangChain ส่งเข้าทีละชิ้น, LlamaIndex ตั้งทีเดียว)
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.llm = Anthropic(model="claude-haiku-5-5", max_tokens=1024)
Settings.node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=40)  # หน่วยเป็น token

# R: Loader -> Document[] -> (split เป็น Node) -> embed -> index
docs = SimpleDirectoryReader(input_files=[str(PDF)]).load_data()  # 1 หน้า = 1 Document
index = VectorStoreIndex.from_documents(docs)

# A+G: query engine = retriever + prompt + llm ในตัวเดียว
engine = index.as_query_engine(similarity_top_k=4)

if __name__ == "__main__":
    for q in (
        "Why does the Transformer scale the dot products by sqrt(d_k)?",
        "Who won the 2022 World Cup?",
    ):
        r = engine.query(q)
        print(f"Q: {q}\nA: {r}")
        for n in r.source_nodes:  # ที่มา + score
            print(f"   - page {n.metadata.get('page_label')} score={n.score:.3f}")
        print()
