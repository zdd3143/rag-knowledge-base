"""阶段 5：混合检索 + 重排。

完整流程：
    问题
      ├─ 向量检索 → Top-50 ┐
      └─ BM25 检索 → Top-50 ┴─ RRF 融合 → Top-20 → cross-encoder 重排 → Top-5

每一步都打印「目标块 #44 排第几」，你能看到它一路升上来。
"""
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from stage4_bm25 import BM25

TARGET = 44          # 正确答案所在的块（阶段 4 诊断出来的）
QUESTION = "中国石油2023年的营业收入是多少？"


def rank_of(results, target=TARGET):
    """目标块在结果里排第几（从 1 开始）；没出现返回 None"""
    ids = [d for d, _ in results]
    return ids.index(target) + 1 if target in ids else None


def show(title, results, chunks, top=5):
    pos = rank_of(results)
    print(f"\n【{title}】目标块 #{TARGET} 排名：{pos if pos else '未出现'}")
    for i, (doc_id, score) in enumerate(results[:top], 1):
        mark = "   ← 目标" if doc_id == TARGET else ""
        text = chunks[doc_id]["text"][:58].replace("\n", " ")
        print(f"  {i}. [{score:.4f}] #{doc_id}: {text}...{mark}")


def main():
    chunks = json.loads(Path("index/chunks.json").read_text(encoding="utf-8"))
    texts = [c["text"] for c in chunks]
    vecs = np.load("index/vectors.npy")
    print(f"语料 {len(texts)} 块")

    # ---------- 准备两路检索器 ----------
    print("\n加载 embedding 模型…")
    model = SentenceTransformer("BAAI/bge-m3")
    print("建立 BM25 索引…")
    bm25 = BM25(texts)
    print("就绪")

    # ---------- 第一路：向量检索 ----------
    q_vec = model.encode(QUESTION, normalize_embeddings=True)
    scores = vecs @ q_vec
    vec_results = [(int(i), float(scores[i])) for i in np.argsort(-scores)[:50]]
    show("向量检索", vec_results, chunks)

    # ---------- 第二路：BM25 ----------
    bm25_results = bm25.search(QUESTION, top_k=50)
    show("BM25 检索", bm25_results, chunks)

    # ---------- 融合：RRF ----------
    K = 60
    fused = {}
    for ranking in (vec_results, bm25_results):
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (K + rank)
    rrf_results = sorted(fused.items(), key=lambda x: -x[1])[:50]
    show("RRF 融合", rrf_results, chunks)

    # ---------- 重排：cross-encoder ----------
    print("\n加载重排模型（第一次会下载约 2GB，耐心等）…")

    candidates = rrf_results[:20]        # 只对前 20 条重排，控制耗时
    pairs = [(QUESTION, texts[d]) for d, _ in candidates]

    print("\n【诊断】RRF 送进重排的前 5 条，原文是什么")
    for i, (d, s) in enumerate(candidates[:5], 1):
      print(f"  {i}. RRF分={s:.5f}  #{d}")
      print(f"     {texts[d][:160]!r}")

    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")


    t0 = time.perf_counter()
    ce_scores = reranker.predict(pairs)
    cost = time.perf_counter() - t0

    reranked = sorted(
        [(d, float(s)) for (d, _), s in zip(candidates, ce_scores)],
        key=lambda x: -x[1])
    show("cross-encoder 重排", reranked, chunks)
    print(f"  （重排 20 条耗时 {cost:.2f} 秒）")

    # ---------- 总结 ----------
    print("\n" + "=" * 62)
    print(f"块 #{TARGET} 的排名变化：")
    print(f"  ① 向量检索      : {rank_of(vec_results) or '未出现'}")
    print(f"  ② BM25          : {rank_of(bm25_results) or '未出现'}")
    print(f"  ③ RRF 融合      : {rank_of(rrf_results) or '未出现'}")
    print(f"  ④ 重排后        : {rank_of(reranked) or '未出现'}")


if __name__ == "__main__":
    main()