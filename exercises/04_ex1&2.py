"""แบบฝึกหัด 4.1 ข้อ 1-2: เทียบ chunk_size และ k"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

PDF = Path(__file__).parent / "data" / "attention.pdf"  # "Attention Is All You Need" (arXiv 1706.03762)
QUERY = "Why does the Transformer scale the dot products by sqrt(d_k)?"


class MiniLM(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


docs = PyPDFLoader(str(PDF)).load()
emb = MiniLM()
print(f"โหลดได้ {len(docs)} Document, ตัวอักษรรวม {sum(len(d.page_content) for d in docs)}\n")

# ข้อ 1: chunk_size 200 vs 1500 (overlap = 1/6 ของ size)
for size in (200, 1500):
    chunks = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=size // 6).split_documents(docs)
    lens = [len(c.page_content) for c in chunks]
    print(f"== chunk_size={size}: {len(chunks)} chunks, ยาวเฉลี่ย {sum(lens)//len(lens)}, max {max(lens)}")
    print(f"   chunk แรก: {chunks[0].page_content[:120]!r}")

# ข้อ 2: k=1 vs k=5 (ใช้ chunk_size=600 คงที่)
chunks = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100).split_documents(docs)
store = InMemoryVectorStore.from_documents(chunks, emb)
for k in (1, 5):
    print(f"\n== k={k}, query={QUERY!r}")
    for d in store.as_retriever(search_kwargs={"k": k}).invoke(QUERY):
        print(f"   [p{d.metadata.get('page')}] {d.page_content[:100]!r}")
