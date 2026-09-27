"""看重排后排名前几的【完整原文】，搞清楚它们为什么得分高。"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from stage4_bm25 import BM25

QUESTION = "中国石油2023年的营业收入是多少？"
TARGET = 44

chunks = json.loads(Path("index/chunks.json").read_text(encoding="utf-8"))
texts = [c["text"] for c in chunks]
vecs = np.load("index/vectors.npy")

model = SentenceTransformer("BAAI/bge-m3")
bm25 = BM25(texts)

# 两路检索 + RRF（和 stage5 一样的流程）
q = model.encode(QUESTION, normalize_embeddings=True)
scores = vecs @ q
vec_results = [(int(i), float(scores[i])) for i in np.argsort(-scores)[:50]]
bm25_results = bm25.search(QUESTION, top_k=50)

fused = {}
for ranking in (vec_results, bm25_results):
    for rank, (d, _) in enumerate(ranking, start=1):
        fused[d] = fused.get(d, 0.0) + 1.0 / (60 + rank)
rrf = sorted(fused.items(), key=lambda x: -x[1])[:20]

# 重排
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
ce = reranker.predict([(QUESTION, texts[d]) for d, _ in rrf])
ranked = sorted([(d, float(s)) for (d, _), s in zip(rrf, ce)],
                key=lambda x: -x[1])

print("=" * 72)
print("重排后完整排名（含每块完整原文）")
print("=" * 72)
for i, (d, s) in enumerate(ranked[:6], 1):
    tag = "   ★★★ 正确答案在这里" if d == TARGET else ""
    print(f"\n[{i}] 分数 {s:.4f} | 块 #{d} | 长度 {len(texts[d])} 字{tag}")
    print("-" * 72)
    print(texts[d])