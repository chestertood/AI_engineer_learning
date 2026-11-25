"""แบบฝึกหัด 4.1 ข้อ 3: ต่อ retriever เข้ากับ Claude ด้วย LCEL (RAG ครบ R+A+G)

ต้องตั้ง ANTHROPIC_API_KEY ก่อนรัน
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

load_dotenv(Path(__file__).parent.parent / ".env")  # อ่าน ANTHROPIC_API_KEY จาก .env

PDF = Path(__file__).parent / "data" / "attention.pdf"
MODEL = "claude-opus-5-5"  # เปลี่ยนเป็น "claude-haiku-5-5" ถ้าอยากถูกลง


class MiniLM(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


# R: index + retrieve
docs = PyPDFLoader(str(PDF)).load()
chunks = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100).split_documents(docs)
retriever = InMemoryVectorStore.from_documents(chunks, MiniLM()).as_retriever(search_kwargs={"k": 4})

# A: prompt, G: llm
prompt = ChatPromptTemplate.from_template(
    "Answer using only the context. If the context does not contain the answer, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)
llm = ChatAnthropic(model=MODEL, max_tokens=1024)


def fmt(docs):
    return "\n\n".join(f"[page {d.metadata.get('page')}] {d.page_content}" for d in docs)


chain = (
    {"context": retriever | fmt, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

if __name__ == "__main__":
    for q in (
        "Why does the Transformer scale the dot products by sqrt(d_k)?",
        "What BLEU score did the big Transformer get on WMT 2014 English-to-German?",
        "Who won the 2022 World Cup?",  # ไม่อยู่ใน paper -> ควรตอบว่าไม่รู้
    ):
        print(f"Q: {q}\nA: {chain.invoke(q)}\n")
