# บทที่ 4.2 LlamaIndex

LlamaIndex = เฟรมเวิร์กที่ **โฟกัสข้อมูล/RAG** (LangChain โฟกัสต่อ chain ทั่วไป) ชิ้นส่วนเหมือนกัน แต่ชื่อต่าง และมี shortcut สูงกว่า

## แผนที่ชื่อ: LangChain -> LlamaIndex

| LangChain | LlamaIndex | หมายเหตุ |
|---|---|---|
| `Document` | `Document` | มี `text` + `metadata` |
| Loader (`PyPDFLoader`) | Reader (`SimpleDirectoryReader`) | เลือก reader ตามนามสกุลไฟล์เอง |
| Splitter -> chunk | NodeParser (`SentenceSplitter`) -> **Node** | chunk เรียกว่า Node, `chunk_size` นับเป็น **token** (LangChain นับตัวอักษร) |
| `Embeddings` | `embed_model` | |
| `VectorStore` | `VectorStoreIndex` | `from_documents()` ทำ split+embed+เก็บให้ |
| `as_retriever()` | `index.as_retriever()` | |
| LCEL chain (retriever \| prompt \| llm) | `index.as_query_engine()` | **บรรทัดเดียว** ได้ R+A+G |

## โค้ดหลัก (รันผ่านแล้ว: `exercises/05_llamaindex.py`)

```python
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.llm = Anthropic(model="claude-haiku-5-5")

docs = SimpleDirectoryReader(input_files=["x.pdf"]).load_data()
index = VectorStoreIndex.from_documents(docs)
engine = index.as_query_engine(similarity_top_k=4)
r = engine.query("...")
print(r)               # คำตอบ
r.source_nodes         # chunk ที่ใช้ + score + metadata (ที่มา)
```

## ต่างจาก LangChain ตรงไหน
- **Settings** = ค่า global (llm, embed, chunk) ตั้งครั้งเดียว. LangChain ส่งเข้าแต่ละชิ้นเอง
- **Query engine** ซ่อน prompt+retriever+llm. สะดวก แต่ปรับยากกว่า (แก้ prompt ต้องใช้ `update_prompts`)
- **response.source_nodes** ได้ที่มา+score ฟรี. LangChain ต้องต่อเอง
- Reader คืน 1 Document ต่อ 1 หน้า PDF ส่วน LangChain ก็ 1 ต่อหน้าเหมือนกัน

## ระวัง
- Windows console cp874 พิมพ์ `√` ไม่ได้ -> รันด้วย `PYTHONIOENCODING=utf-8`
- ต้องลง integration แยก: `llama-index-embeddings-huggingface`, `llama-index-llms-anthropic`, `llama-index-readers-file` (อ่าน PDF)
- ถ้าไม่ตั้ง `Settings.llm/embed_model` มันจะ default ไป OpenAI -> error ขอ key
- ตอนติดตั้ง numpy ถูกอัพเป็น 2.2.6 -> tensorflow/mealpy ในเครื่อง warn (ไม่เกี่ยวกับ RAG)

## แบบฝึกหัด
1. เปลี่ยน `chunk_size` 128 vs 1024 ดู `source_nodes` เปลี่ยนยังไง
2. ใช้ `index.as_retriever(similarity_top_k=3).retrieve(q)` แยกส่วน R ออกมาดู (ไม่เรียก LLM)
3. ถามคำถามนอก paper ดู score ต่ำ (ตัวอย่างวันนี้: 0.08 vs 0.40)

## คำถามเช็กความเข้าใจ (รอคำตอบ)
ทำไม LlamaIndex ใช้ `Settings` global แต่ LangChain ส่ง `llm`/`embeddings` เข้าแต่ละชิ้น? แต่ละแบบข้อดี/เสียอะไร?
