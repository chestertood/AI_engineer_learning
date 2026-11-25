# บทที่ 4.3 เทียบ LangChain vs LlamaIndex

สรุปหลักก่อน: **ทั้งคู่ทำ RAG ได้เหมือนกัน ชิ้นส่วนเหมือนกัน ต่างที่ปรัชญาและระดับ abstraction**

| | LangChain | LlamaIndex |
|---|---|---|
| จุดยืน | **กล่องเครื่องมือทั่วไป** ต่อ LLM กับอะไรก็ได้ (chain, tool, agent) | **เฟรมเวิร์กข้อมูล** เน้น ingest -> index -> retrieve -> query (RAG เป็นพระเอก) |
| ระดับ abstraction | ต่ำ-กลาง: ต่อชิ้นเอง (`\|`) เห็นทุกขั้น | กลาง-สูง: `as_query_engine()` ซ่อนทั้ง R+A+G |
| บรรทัดจน RAG | ~10-15 (loader, splitter, store, retriever, prompt, llm, parser) | ~5 (reader, index, engine, query) |
| ตั้งค่า LLM/embedding | **ส่งเข้าแต่ละชิ้น** (explicit) | **`Settings` global** ตั้งครั้งเดียว (implicit) |
| หน่วย chunk | ตัวอักษร (`chunk_size=1000` = 1000 ตัวอักษร) | **token** (`chunk_size=256` = 256 token) |
| chunk เรียกว่า | `Document` (ใช้ซ้ำทั้งก่อน/หลังตัด) | `Node` (หลังตัด, มี relationship ถึง node ข้างเคียง) |
| ที่มา/score | ต้องต่อเอง (เก็บ metadata ออกมา) | **ฟรี** `response.source_nodes` |
| ปรับ prompt | แก้ template ตรงๆ ง่าย | ต้อง `update_prompts` / ประกอบเอง (query engine ปิดอยู่) |
| Agent/workflow | LangGraph (กราฟ, state) | Agent + Workflows (event-driven) |
| Data connectors | `langchain-community` (บางส่วน sunset) | LlamaHub (เน้นแหล่งข้อมูลเยอะ) |
| Observability | LangSmith | Callback + เครื่องมือภายนอก |
| ข้อเสียร่วม | API เปลี่ยนบ่อย, abstraction หนา debug ยาก | เหมือนกัน |

## RAG เดียวกัน เทียบข้างกัน

```python
# LangChain: ต่อเอง เห็นทุกชิ้น
chain = RunnableLambda(get_context) | prompt | llm | StrOutputParser()
answer = chain.invoke({...})

# LlamaIndex: ตั้ง Settings ครั้งเดียว แล้ว
index  = VectorStoreIndex.from_documents(docs)
answer = index.as_query_engine().query("...")
```
ไฟล์จริงในเครื่อง: `Lang_chain.py`, `exercises/04_ex3.py` (LangChain) vs `exercises/05_llamaindex.py` (LlamaIndex)

## คำตอบคำถามค้างจาก 4.2: ทำไม `Settings` global vs ส่งเข้าแต่ละชิ้น

| | Global (`Settings`) | Explicit (ส่งเข้าชิ้น) |
|---|---|---|
| ข้อดี | โค้ดสั้น ตั้งครั้งเดียว ทุกชิ้นใช้ตรงกัน (embed ตอนสร้าง DB = embed ตอนค้น) | เห็นชัดว่าชิ้นไหนใช้อะไร, ชิ้นต่างกันใช้โมเดลต่างกันได้, ทดสอบง่าย (ใส่ mock) |
| ข้อเสีย | **ซ่อนค่า**: ถ้าลืมตั้ง จะ default ไป OpenAI แล้ว error ขอ key (เราเจอมาแล้ว), ค่าเป็น state ร่วมกันทั้งโปรแกรม เปลี่ยนแล้วกระทบทุกที่ | โค้ดยาวขึ้น ส่งซ้ำหลายจุด ลืมส่ง/ส่งไม่ตรงกันได้ (embed ต่างโมเดลตอนสร้าง vs ค้น = ค้นไม่เจอ) |
| เข้ากับ | แอปที่ค่าเดียวใช้ทั้งระบบ | งานที่หลายโมเดล/หลาย config ในโปรแกรมเดียว |

LlamaIndex ก็ override เป็นรายชิ้นได้ (`as_query_engine(llm=...)`) และ LangChain ก็ตั้ง default ได้ผ่าน config. ต่างกันที่ **ค่าเริ่มต้นของปรัชญา** ไม่ใช่ข้อจำกัดตายตัว.

## เลือกอะไรเมื่อไหร่

- **งานหลักคือ RAG กับเอกสารเยอะ ต้องการเริ่มเร็ว + ได้ที่มา** -> LlamaIndex
- **ต้องต่อ LLM กับเครื่องมือ/API/หลายขั้นที่ไม่ใช่ RAG อย่างเดียว, agent ซับซ้อน, คุมทุกขั้น** -> LangChain (+ LangGraph)
- **ต้องการคุมละเอียดมาก/ไม่อยากพึ่ง API ที่เปลี่ยนบ่อย** -> เขียนเอง (`RAG_by_myself.py`) ก็พอ ชิ้นส่วนไม่กี่ร้อยบรรทัด
- ผสมกันได้ เช่นใช้ LlamaIndex ทำ retrieval แล้วเอาเข้า LangChain/LangGraph เป็น tool

> กฎจำง่าย: ส่วนไหนเป็น "ข้อมูล" -> LlamaIndex เด่น. ส่วนไหนเป็น "การไหลของงาน/agent" -> LangChain เด่น.

## ข้อสังเกตจากที่เราลงมือจริง
- `RAG_by_myself.py` (เขียนเอง) ทำให้เข้าใจว่าสองเฟรมเวิร์กแค่ห่อ chunk/embed/retrieve/prompt/LLM ที่เราเขียนเองมาแล้ว -> ไม่เป็นกล่องดำ
- chunk_size คนละหน่วย: LangChain 1000 ตัวอักษร ~ 250 token ไม่เท่ากับ LlamaIndex 1000 token. เทียบผลต้องปรับหน่วยให้ตรงก่อน
- ผลต่างของคำตอบมักมาจาก **chunk / K / โมเดล** มากกว่าตัวเฟรมเวิร์ก

## แบบฝึกหัด
1. ถามคำถามเดียวกัน ("Why does the Transformer scale the dot products by sqrt(d_k)?") กับ `04_ex3.py` และ `05_llamaindex.py` เทียบ chunk ที่ดึงมา (ตั้ง chunk ให้ ~ใกล้กัน: ตัวอักษร 1000 ~ token 250) ใครดึงตรงกว่า
2. ใน LlamaIndex ใช้ `index.as_retriever(similarity_top_k=3).retrieve(q)` เทียบกับ `db.as_retriever(k=3).invoke(q)` ของ LangChain: ผลลัพธ์/score หน้าตาต่างกันยังไง
3. (ท้าทาย) เปลี่ยน LLM จาก Claude เป็นโมเดลอื่น ในแต่ละเฟรมเวิร์ก แก้กี่จุด

## คำถามเช็กความเข้าใจ 4.3 (รอคำตอบ)
งานของคุณคือแชตบอตตอบจากเอกสารบริษัท 500 ไฟล์ + ต้องแสดงว่าตอบมาจากหน้าไหน. เลือกเฟรมเวิร์กไหนเริ่ม และเพราะอะไร? (ตอบ 2-3 บรรทัด)
