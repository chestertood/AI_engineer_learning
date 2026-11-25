"""Retriever metrics ลงมือจริง (ไม่ต้องใช้ LLM): ใช้ label ที่คนตัดสินไว้แล้วแทน LLM-judge"""


def context_precision_at_k(relevance):
    """relevance = [1,0,0,...] ตามลำดับอันดับที่ retriever คืนมา (1 = chunk เกี่ยว)"""
    hits, total = 0, 0.0
    for k, v in enumerate(relevance, start=1):
        hits += v
        total += (hits / k) * v  # Precision@k x v_k
    return total / hits if hits else 0.0


def context_recall(claims_supported):
    """claims_supported = [True, True, True, False] ต่อ claim ในเฉลย"""
    return sum(claims_supported) / len(claims_supported)


def entities_recall(ref_entities, ctx_entities):
    ref = set(ref_entities)
    return len(ref & set(ctx_entities)) / len(ref)


def context_relevancy(statements_relevant):
    """สัดส่วน statement ใน context ที่เกี่ยวกับคำถาม"""
    return sum(statements_relevant) / len(statements_relevant)


if __name__ == "__main__":
    # ตัวอย่างจาก doc: เกี่ยวอยู่อันดับ 1 -> 1.0
    assert context_precision_at_k([1, 0, 0]) == 1.0
    # เกี่ยวอยู่อันดับท้าย -> ต่ำลง (1/3)
    print("relevant @ rank 3  :", round(context_precision_at_k([0, 0, 1]), 3))
    # มี 2 chunk เกี่ยว: เรียงดี vs เรียงแย่
    print("[1,1,0,0]          :", round(context_precision_at_k([1, 1, 0, 0]), 3))  # 1.0
    print("[0,0,1,1]          :", round(context_precision_at_k([0, 0, 1, 1]), 3))  # ~0.417
    print("[1,0,1,0]          :", round(context_precision_at_k([1, 0, 1, 0]), 3))  # ~0.833
    print("none relevant      :", context_precision_at_k([0, 0, 0]))

    assert context_recall([True, True, True, False]) == 0.75
    assert round(entities_recall(["Brazil", "Brasília", "April 21, 1960"], ["Brasília", "Brazil"]), 2) == 0.67
    assert round(context_relevancy([True, False, True]), 2) == 0.67
    print("ok")
