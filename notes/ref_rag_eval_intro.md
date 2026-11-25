# อ้างอิง: RAG Evaluation Metrics (เก็บไว้รอบท 16)

ที่มา: https://github.com/KalyanKS-NLP/rag-zero-to-hero-guide/blob/main/RAG%20Evaluation/rag_eval_intro.md
สถานะ: อ่านแล้ว สรุปไว้ ยังไม่ได้เรียน/ลงมือ (ตรงกับบท 16 RAG Evaluation & Monitoring)

## แนวคิดหลัก
RAG มี 2 ส่วน -> **ประเมินแยกกัน** เพราะพังคนละแบบ
- **Retriever** ดึง context ที่เกี่ยวกับคำถาม. ถ้าดึงผิด/ไม่ครบ LLM ก็ตอบผิดได้ ต่อให้ LLM เก่ง
- **Generator (LLM)** ตอบจาก query + context. ถ้าแย่ ก็ตอบมั่ว/ไม่ตรงได้ ต่อให้ context สมบูรณ์

## Type ของ metric
- **Reference Dependent** = ต้องมีคำตอบอ้างอิง (ground truth) ไว้เทียบ
- **Reference Free** = ไม่ต้องมีเฉลย ใช้แค่ query/context/response (ใช้ได้กับข้อมูลจริงที่ไม่มีเฉลย)

## Retriever
| Metric | ใช้ | Type | วัดอะไร |
|---|---|---|---|
| Context Precision | Context, Reference | ต้องมีเฉลย | **จัดอันดับ** chunk ที่เกี่ยวขึ้นบนได้ดีไหม |
| Context Recall | Context, Reference | ต้องมีเฉลย | ดึงข้อมูลที่ **จำเป็นต่อการตอบ** ได้ครบไหม |
| Context Entities Recall | Context, Reference | ต้องมีเฉลย | ดึง **entity** (ชื่อ/สถานที่/ฯลฯ) ที่อยู่ในคำตอบเฉลยได้ครบไหม |
| Context Relevancy | Context, Query | ไม่ต้องมีเฉลย | context ที่ดึงมา **เกี่ยวกับคำถาม** แค่ไหน |

## Generator
| Metric | ใช้ | Type | วัดอะไร |
|---|---|---|---|
| Faithfulness | Context, Response | ไม่ต้องมีเฉลย | คำตอบ **สอดคล้องกับ context** ที่ได้แค่ไหน |
| Hallucination | Context, Response | ไม่ต้องมีเฉลย | คำตอบ **ขัด/เกินจาก context** แค่ไหน (ด้านกลับของ faithfulness) |
| Response Relevancy | Response, Query | ไม่ต้องมีเฉลย | คำตอบ **ตรงคำถาม** แค่ไหน |
| Relevant Noise Sensitivity | Context, Response, Reference | ต้องมีเฉลย | ไวต่อ noise (claim ผิด) ใน context ที่ **เกี่ยวข้อง** แค่ไหน |
| Irrelevant Noise Sensitivity | Context, Response, Reference | ต้องมีเฉลย | ไวต่อ noise ใน context ที่ **ไม่เกี่ยวข้อง** แค่ไหน |

## เชื่อมกับสิ่งที่เจอจริงตอนเรียน (ตัวอย่างเข้าใจง่าย)
- **YouTube RAG ตอบ "not available"** (chunk คำตอบอยู่อันดับ 5 แต่ K=2) = ปัญหา **Context Recall / Precision** ของ retriever (ไม่ใช่ LLM)
- **Prompt "ตอบจาก context เท่านั้น"** = พยายามกด **Hallucination** ให้ต่ำ / เพิ่ม **Faithfulness**
- **Evaluator agent ใน Agentic RAG** (yes/no ว่า context พอไหม) = การประเมิน **Context Relevancy** แบบ real-time ใน pipeline

## เครื่องมือที่น่าจะใช้ตอนลงมือ (ยังไม่ได้ตรวจ ต้องเช็กเอกสารตอนเรียน)
ชื่อ metric ชุดนี้คล้ายกับของ **Ragas** (context precision/recall/entities recall, faithfulness, response relevancy, noise sensitivity). ตอนบท 16 ค่อยเช็กว่าใช้ Ragas หรือเครื่องมืออื่น.

## ข้อควรจำ
- ประเมิน retriever ก่อนเสมอ เพราะ generator ดีแค่ไหนก็ไม่ช่วยถ้า context ผิด
- metric แบบไม่ต้องมีเฉลย ใช้ monitoring ใน production ได้. แบบต้องมีเฉลย ต้องเตรียมชุดคำถาม+คำตอบเอง
