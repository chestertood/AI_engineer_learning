"""RAG with LangChain (ตาม notebook RAG_with_LangChain.ipynb ปรับให้ใช้ MiniLM + Claude)

7 step: extract -> chunk -> vector store -> retrieve -> context -> chain -> run
ต้อง: pip install langchain-huggingface langchain-chroma langchain-anthropic pypdf
"""
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")  # อ่าน ANTHROPIC_API_KEY

# ค่าที่ปรับได้ (เปลี่ยน CHUNK_* หรือ embedding ต้องลบโฟลเดอร์ db ก่อน; K เปลี่ยนแล้วรันได้เลย)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
K = 2
MODEL = "claude-haiku-5-5"
DB_PATH = str(HERE / "db" / "chroma_langchain")  # แยกจาก db ของ RAG_by_myself.py

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Step 1: Extract
def pdf_extract(pdf_path: str) -> List[Document]:
    return PyPDFLoader(pdf_path).load()  # 1 Document ต่อ 1 หน้า


# Step 2: Chunk
def pdf_chunk(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)


# Step 3: Vector store (โหลดของเดิมถ้ามีข้อมูล ไม่งั้นสร้างใหม่)
def get_vector_store(pdf_path: str, db_path: str) -> Chroma:
    db = Chroma(persist_directory=db_path, embedding_function=embedding_model)
    if db._collection.count() == 0:  # เช็กจากข้อมูลจริง กันโฟลเดอร์ว่างหลอก
        print("Creating new vector store...")
        db.add_documents(pdf_chunk(pdf_extract(pdf_path)))
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
    db = get_vector_store(inputs["pdf_path"], inputs["db_path"])
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
    # pdf_path = str(HERE / "test.pdf")
    # for query in ["What is the transformer architecture?", "What is self-attention?"]:
    #     answer = rag_chain.invoke({"pdf_path": pdf_path, "query": query, "db_path": DB_PATH})
    #     print(f"Query: {query}\nGenerated answer: {answer}\n")

    pdf_text = pdf_extract("attention_is_all_you_need.pdf")
    print(f"number of documents = {len(pdf_text)}")