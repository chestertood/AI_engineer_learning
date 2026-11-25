"""YouTube Video RAG (ตาม notebook YouTube_Video_RAG.ipynb ปรับให้ใช้ MiniLM + Claude เหมือน Lang_chain.py)

7 step: extract transcript -> chunk -> vector store -> retrieve -> context -> chain -> run
ต้อง: pip install langchain-huggingface langchain-chroma langchain-anthropic langchain-community youtube-transcript-api
"""
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")  # อ่าน ANTHROPIC_API_KEY

# ถ้าเปลี่ยน VIDEO_URL ต้องลบโฟลเดอร์ db/chroma_yt ก่อน (DB ผูกกับ path ไม่ใช่ URL)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
K = 5  # transcript มีคำผิด (fery/faery/fairy) + MiniLM อ่อน -> chunk คำตอบอยู่อันดับ 5
MODEL = "claude-haiku-5-5"
DB_PATH = str(HERE / "db" / "chroma_yt")  # แยกจาก db ของไฟล์อื่น

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Step 1: Extract transcript (YoutubeLoader คืน 1 Document = transcript ทั้งคลิป)
def yt_transcript(video_url: str) -> List[Document]:
    print("YouTube video transcript is extracted...")
    return YoutubeLoader.from_youtube_url(video_url).load()


# Step 2: Chunk
def yt_chunk(transcript: List[Document]) -> List[Document]:
    print("Transcript is chunked...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(transcript)


# Step 3: Vector store (โหลดของเดิมถ้ามีข้อมูล ไม่งั้นสร้างใหม่)
def get_vector_store(video_url: str, db_path: str) -> Chroma:
    db = Chroma(persist_directory=db_path, embedding_function=embedding_model)
    if db._collection.count() == 0:
        print("Creating new vector store...")
        db.add_documents(yt_chunk(yt_transcript(video_url)))
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
    db = get_vector_store(inputs["video_url"], inputs["db_path"])
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
    video_url = "https://www.youtube.com/watch?v=d4IyR-kl_mY"
    query = "passenger ferry service is resumed from which place?"
    answer = rag_chain.invoke({"video_url": video_url, "query": query, "db_path": DB_PATH})
    print(f"Query: {query}\nGenerated answer: {answer}")
