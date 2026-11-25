# บทที่ 4.1 LangChain

LangChain = ชุดชิ้นส่วนมาตรฐานที่ต่อกันได้ แบ่ง 2 ครึ่ง

## ครึ่ง 1: Retrieval (Index + Retrieve)

```
PDF -Loader-> Document[] -Splitter-> chunks -Embeddings-> VectorStore -> Retriever
```

| ชิ้น | คืออะไร | จุดที่ต้องรู้ |
|---|---|---|
| `Document` | `page_content` + `metadata` | metadata (หน้า, ชื่อไฟล์) ใช้อ้างอิงที่มา |
| Loader | ไฟล์ -> `Document[]` | ทุกตัวเรียก `.load()` เหมือนกัน |
| Splitter | ตัด Document เป็น chunk | `chunk_size`, `chunk_overlap` กันประโยคขาด |
| `Embeddings` | ข้อความ -> vector | ต้องมี `embed_documents`, `embed_query` |
| `VectorStore` | เก็บ vector + ค้น | `from_documents()` embed+เก็บให้ |
| Retriever | คำถาม -> Document[] | `store.as_retriever(search_kwargs={"k": 3})` |

ทุกชิ้นมี interface เดียวกัน -> สลับ InMemory เป็น Chroma/FAISS แก้บรรทัดเดียว

เทียบกับที่เขียนเองใน `test_Rag.py`:
pdf_chunks()=PyPDFLoader, chunk_text()=RecursiveCharacterTextSplitter, embed()=Embeddings subclass, vecs+cosine=InMemoryVectorStore, retrieve()=as_retriever

## ครึ่ง 2: Generation (Augment + Generate) ด้วย LCEL

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_anthropic import ChatAnthropic

prompt = ChatPromptTemplate.from_template(
    "ตอบจาก context เท่านั้น ถ้าไม่มีให้บอกว่าไม่รู้\n\n"
    "Context:\n{context}\n\nคำถาม: {question}"
)
llm = ChatAnthropic(model="claude-haiku-5-5")

def fmt(docs):
    return "\n\n".join(d.page_content for d in docs)

chain = (
    {"context": retriever | fmt, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
chain.invoke("มะเขือเทศต้องการ EC เท่าไหร่?")
```

(ยังไม่ได้รันจริง ต้องมี ANTHROPIC_API_KEY)

อ่านซ้าย->ขวา:
1. dict แรกทำ 2 อย่างพร้อมกัน: คำถาม -> retriever -> fmt = `context`; คำถามเดิมส่งต่อเป็น `question`
2. `prompt` เติมสองช่องลง template
3. `llm` ตอบ -> `AIMessage`
4. `StrOutputParser` ดึงเหลือ string

## Runnable
retriever, prompt, llm, parser เป็น Runnable หมด: `.invoke(x)`, `.batch([..])`, `.stream(x)`. `|` = ต่อ Runnable

## ระวัง
- `RetrievalQA`, `LLMChain` เลิกใช้แล้ว -> ใช้ LCEL
- `langchain-community` ถูก sunset (DeprecationWarning)
- API เปลี่ยนบ่อย ตรวจ doc เวอร์ชันที่ติดตั้ง (langchain 1.x)

## แบบฝึกหัด
1. เปลี่ยน `chunk_size` 200 vs 1500 ดู chunk ที่ได้
2. เปลี่ยน `k` 1 vs 5
3. ต่อ chain ข้างบนเข้ากับ retriever (ต้องใช้ PDF ที่อ่านออก)

ตัวอย่างรันได้: `C:\Users\chest\OneDrive\Desktop\LLM_fertilizer\learn\04_langchain_basics.py` (retriever เท่านั้น ยังไม่มี chain)

## คำถามเช็กความเข้าใจ (รอคำตอบ)
ทำไม dict แรกใน chain ใช้ `RunnablePassthrough()` กับ `question`?

<details><summary>เฉลย</summary>
chain ได้อินพุตเป็นคำถามก้อนเดียว แต่ prompt ต้องการ 2 ช่อง (context, question). dict ทำให้ใช้อินพุตเดียวกันแตกเป็น 2 ทางพร้อมกัน: ทางหนึ่งผ่าน retriever ได้ context, อีกทางส่งคำถามเดิมผ่าน RunnablePassthrough (ส่งค่าเข้า = ออก ไม่แก้) ไปเป็น question.
</details>
