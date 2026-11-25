# อ้างอิง: RAG Retriever Evaluation Metrics (เก็บไว้รอบท 16)

ที่มา: https://github.com/KalyanKS-NLP/rag-zero-to-hero-guide/blob/main/RAG%20Evaluation/rag_retriever_metrics.md
ภาพรวม metric ทั้งหมดดู `notes/ref_rag_eval_intro.md`. ไฟล์นี้เจาะ 4 metric ของ **Retriever**
สถานะ: อ่านแล้ว สรุปไว้ ยังไม่ได้ลงมือ

ทุกตัว: ช่วงคะแนน **0-1** (1 = ดีสุด)

## 1. Context Precision@K: "chunk ที่เกี่ยวขึ้นอันดับบนไหม" (สนใจ **ลำดับ**)
- ต้องมี: Reference (เฉลย) + Retrieved chunks (เรียงตามอันดับ)
- สูตร: `sum( Precision@k x v_k ) / จำนวน chunk ที่เกี่ยวใน top K`
  - `v_k` = 1 ถ้า chunk อันดับ k เกี่ยว, 0 ถ้าไม่เกี่ยว
  - `Precision@k` = (chunk เกี่ยวถึงอันดับ k) / k
- วิธีคิด: ตัดสินทีละ chunk ว่าเกี่ยวไหม -> คำนวณ Precision@k ทุกอันดับ -> รวมเฉพาะอันดับที่เกี่ยว -> หารด้วยจำนวนที่เกี่ยว
- ตัวอย่าง (ทะเลทรายใหญ่สุด): chunk1 เกี่ยว, chunk2-3 ไม่เกี่ยว -> (1.0x1)/1 = **1.0**. ถ้า chunk เกี่ยวอยู่ท้ายสุดคะแนนจะต่ำลง
- 1 = เกี่ยวหมดและเรียงดี. 0 = ไม่มี chunk เกี่ยวเลย

## 2. Context Recall: "ดึงข้อมูลที่ต้องใช้ครบไหม" (สนใจ **ความครบ**)
- ต้องมี: Reference + Retrieved contexts. ใช้ **LLM** ช่วยตัดสิน
- สูตร: `จำนวน claim ในเฉลยที่หาหลักฐานเจอใน context / จำนวน claim ในเฉลยทั้งหมด`
- วิธีคิด: LLM แตกเฉลยเป็น claim ย่อยๆ -> เช็กว่าแต่ละ claim มี context รองรับไหม -> นับ
- ตัวอย่าง (สาเหตุตัดไม้ทำลายป่า): เฉลย 4 ข้อ (logging, agriculture, urbanization, wildfires) context รองรับ 3 -> **0.75** (ขาด wildfires)
- 0 = ดึงผิดหมด

## 3. Context Entities Recall: "ดึง entity (ชื่อ/สถานที่/วันที่) ในเฉลยครบไหม"
- สูตร: `|RCE ∩ RE| / |RE|` (RE = entity ในเฉลย, RCE = entity ใน context)
- วิธีคิด: ดึง entity จากเฉลย + จาก context -> หาส่วนที่ซ้ำ -> หาร
- ตัวอย่าง (เมืองหลวงบราซิล): เฉลยมี 3 entity (Brazil, Brasília, April 21 1960) context มี 2 -> **0.67** (ขาดวันที่)
- เหมาะกับคำถามเชิงข้อเท็จจริง (ชื่อ/ตัวเลข/วันที่) ที่ entity สำคัญ

## 4. Context Relevancy: "context ที่ดึงมาเกี่ยวกับคำถามแค่ไหน" (สนใจ **ขยะปน**)
- ต้องมี: **Query + Retrieved contexts** (ไม่ต้องมีเฉลย = Reference Free)
- สูตร: `จำนวน statement ใน context ที่เกี่ยวกับคำถาม / statement ทั้งหมดใน context`
- วิธีคิด: LLM แตก context เป็น statement -> ตัดสินแต่ละอันว่าเกี่ยวไหม -> นับ
- ตัวอย่าง (ประโยชน์ชาเขียว): 3 statement เกี่ยว 2 (อีกอันเรื่องกาแฟ) -> **0.67**

## จำแบบสั้น
| Metric | ถามว่า | ต้องมีเฉลย | ตกเมื่อ |
|---|---|---|---|
| Context Precision | ของดีอยู่บนๆ ไหม | ใช่ | chunk ที่เกี่ยวถูกฝังอันดับท้าย |
| Context Recall | ได้ครบไหม | ใช่ | ขาดข้อมูลที่ต้องใช้ตอบ |
| Entities Recall | ได้ชื่อ/ตัวเลขครบไหม | ใช่ | ขาด entity สำคัญ |
| Context Relevancy | มีขยะปนไหม | **ไม่** | ดึง chunk ไม่เกี่ยวมาเยอะ |

- **Recall vs Relevancy ตรงข้ามกัน**: เพิ่ม K -> recall ขึ้น แต่ relevancy/precision มักลง (trade-off เดียวกับที่เจอตอนตั้ง K=5 ใน YouTube RAG)
- 3 ตัวแรกต้องมีเฉลย (เตรียมชุดคำถาม+คำตอบเอง). Recall/Relevancy/Entities ใช้ LLM หรือการสกัด entity ช่วยตัดสิน -> ผลขึ้นกับ LLM ที่ใช้เป็นกรรมการ

## เชื่อมกับของที่เจอจริง
- YouTube RAG K=2 พลาด: **Context Recall ต่ำ** (chunk คำตอบไม่ติด) และ **Precision ต่ำ** (chunk ที่เกี่ยวอยู่อันดับ 5)
- เกณฑ์ `similarity_threshold` ของ CrewAI tool = ตัวกรองแบบคล้าย relevancy: ตั้งสูงไปก็ recall ตก

หมายเหตุ: ชื่อ/สูตรตรงกับ Ragas (ฉันเดา ยังไม่ได้ตรวจ) ตอนบท 16 ให้เช็กเอกสารเครื่องมือจริงก่อน.
