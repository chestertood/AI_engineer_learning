from PyPDF2 import PdfReader
from typing import List
import requests
import re
from collections import deque
import chromadb
from chromadb.utils import embedding_functions
from chromadb.api.models import Collection
from pathlib import Path
from dotenv import load_dotenv
import os
import uuid
from typing import Any, List
from typing import Tuple
from litellm import completion

load_dotenv(Path(__file__).parent / ".env")

def text_extract(pdf_path:str) -> str:
    pdf_pages = []
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text = page.extract_text()
            pdf_pages.append(text)
    pdf_text = "\n".join(pdf_pages)
    return pdf_text

def write_pdf(url:str):
    pdf_url = url
    response = requests.get(pdf_url)

    pdf_path = 'test.pdf'
    with open(pdf_path, "wb") as file:
        file.write(response.content)

    return print("complete dowload pdf")

def text_chunk(text:str, max_length: int = 1000) -> List[str]:
    sentences = deque(re.split(r'(?<=[.!?])\s+', text.replace('\n', ' ')))
    chunks = []
    chunk_text = ""
    while sentences:
        sentence = sentences.popleft().strip()

        if sentence:
            if len(chunk_text) + len(sentence) > max_length and chunk_text:
                chunks.append(chunk_text)
                chunk_text = sentence
            else:
                chunk_text += " " + sentence
    if chunk_text:
        chunks.append(chunk_text)
    return chunks

def create_vector_store(db_path: str) -> Collection:
    client = chromadb.PersistentClient(path=db_path)
    embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    db =client.create_collection(
        name="pdf_chunks",
        embedding_function=embeddings
    )

    return db

def insert_chunks_vectordb(chunks: List[str], db: Collection, file_path:str) -> None:
    file_name = os.path.basename(file_path)

    id_list = [str(uuid.uuid4()) for _ in range(len(chunks))]

    metadata_list = [{"chunk": i , "source": file_name} for i in range(len(chunks))]

    batch_size = 40

    for i in range(0, len(chunks), batch_size):
        end_id = min(i + batch_size, len(chunks))

        db.add(
            documents=chunks[i:end_id],
            metadatas = metadata_list[i:end_id],
            ids=id_list[i:end_id]
        )
    print(f"{len(chunks)} chunks added to vector store")



def retrieve_chunks(db: Collection, query: str, n_results: int = 6) -> List[Any]:
    relevant_chunks = db.query(
        query_texts=[query],
        n_results=n_results
    )

    return relevant_chunks


def build_context(relevant_chunks) -> str:
    context = "\n".join(relevant_chunks['documents'][0])
    return context

def get_context(pdf_path: str,query: str, db_path: str) -> Tuple[str, str]:
    if os.path.exists(db_path):
        print("Loading existing vector store...")

        client = chromadb.PersistentClient(path=db_path)

        embeddings = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        db = client.get_collection(name="pdf_chunks", embedding_function=embeddings)
    else:
        print("Creating new vector store...")

        pdf_text = text_extract(pdf_path)

        chunks = text_chunk(pdf_text)

        db = create_vector_store(db_path)

        insert_chunks_vectordb(chunks, db, pdf_path)

    relevant_chunks = retrieve_chunks(db, query)

    context = build_context(relevant_chunks)

    return context, query

def get_prompt(context: str, query: str) -> str:
    rag_prompt = f""" You are an AI model trained for question answering. You should answer the
    given question based on the given context only.
    Question : {query}
    \n
    Context : {context}
    \n
    If the answer is not present in the given context, respond as: The answer to this question is not available
    in the provided content.
    """
    return rag_prompt

def get_response(rag_prompt: str) -> str : 
    model = "anthropic/claude-sonnet-5-5"

    messages = [{"role": "user", "content": rag_prompt}]

    response = completion(
        model=model,
        messages=messages,
    )

    answer = response.choices[0].message.content
    return answer

def rag_pipeline(pdf_path: str, query: str, db_path: str) -> str:
    context, query = get_context(pdf_path, query, db_path)

    rag_prompt = get_prompt(context, query)

    response = get_response(rag_prompt)

    return response


    

if __name__ == "__main__":
    current_dir = "."
    persistent_directory = os.path.join(current_dir, "db", "chroma_db_pdf")

    pdf_path = "test.pdf"

    query = "What is the transformer architecture?"

    answer = rag_pipeline(pdf_path, query, persistent_directory)

    print(f"Query: {query}")
    print(f"Generated Answer: {answer}")

    query2 = "What is self-attention?"

    answer2 = rag_pipeline(pdf_path, query2, persistent_directory)

    print(f"Query2: {query2}")
    print(f"Generated Answer: {answer2}") 