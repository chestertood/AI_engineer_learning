"""Web Page RAG (ตาม notebook Web_Page_RAG.ipynb ปรับให้ใช้ MiniLM + Claude เหมือน Lang_chain.py)

7 step: extract -> chunk -> vector store -> retrieve -> context -> chain -> run
ต้อง: pip install langchain-huggingface langchain-chroma langchain-anthropic langchain-community beautifulsoup4
"""
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")  # อ่าน ANTHROPIC_API_KEY

# ถ้าเปลี่ยน PAGE_URL ต้องลบโฟลเดอร์ db ก่อน (DB ผูกกับ path ไม่ใช่ URL)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
K = 2
MODEL = "claude-haiku-5-5"
DB_PATH = str(HERE / "db" / "chroma_web")  # แยกจาก db ของไฟล์อื่น

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Step 1: Extract (WebBaseLoader คืน 1 Document ต่อ 1 URL ทั้งหน้า)
def wp_text(page_url: str) -> List[Document]:
    print("Web page text is extracted...")
    return WebBaseLoader(page_url).load()


# Step 2: Chunk
def wp_chunk(docs: List[Document]) -> List[Document]:
    print("Web page text is chunked...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)


# Step 3: Vector store (โหลดของเดิมถ้ามีข้อมูล ไม่งั้นสร้างใหม่)
def get_vector_store(page_url: str, db_path: str) -> Chroma:
    db = Chroma(persist_directory=db_path, embedding_function=embedding_model)
    if db._collection.count() == 0:
        print("Creating new vector store...")
        db.add_documents(wp_chunk(wp_text(page_url)))
    else:
        print("Loading existing vector store...")
    return db


# Step 4: Retrieve
def retrieve_context(db: Chroma, query: str) -> List[Document]:
    return db.as_retriever(search_type="similarity", search_kwargs={"k": K}).invoke(query)


# Step 5: Build context
def build_context(chunks: List[Document]) -> str:
    return "\n\n".join(c.page_content for c in chunks)


def get_context(inputs: Dict[str, str]) -> Dict[str, str]:
    db = get_vector_store(inputs["page_url"], inputs["db_path"])
    context = build_context(retrieve_context(db, inputs["query"]))
    return {"context": context, "query": inputs["query"]}


# Step 6: Chain = get_context | prompt | llm | parser
template = """You are an AI model trained for question answering. You should answer the
given question based on the given context only.
Question : {query}

Context : {context}

If the answer is not present in the given context, respond as: The answer to this question is not available
in the provided content.
"""
rag_prompt = ChatPromptTemplate.from_template(template)
llm = ChatAnthropic(model=MODEL, max_tokens=1024)
rag_chain = RunnableLambda(get_context) | rag_prompt | llm | StrOutputParser()


# Step 7: Run
if __name__ == "__main__":
    page_url = "https://x.ai/blog/grok-2"
    query = "What is Grok 2?"
    answer = rag_chain.invoke({"page_url": page_url, "query": query, "db_path": DB_PATH})
    print(f"Query: {query}\nGenerated answer: {answer}")
