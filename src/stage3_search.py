"""加载索引并检索。对比「纯 Python 循环」和「numpy 矩阵」两种实现。"""
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path("index")


def search_python(q, vecs, top_k=5):
    """纯 Python 循环：一条一条算点积。直观，但慢。"""
    scored = []
    for i, v in enumerate(vecs):
        dot = 0.0
        for a, b in zip(q, v):
            dot += float(a) * float(b)
        scored.append((dot, i))
    scored.sort(reverse=True)
    return scored[:top_k]


def search_numpy(q, vecs, top_k=5):
    """矩阵乘法：一次算出所有相似度。"""
    scores = vecs @ q                      # (N, dim) @ (dim,) -> (N,)
    order = np.argsort(-scores)[:top_k]    # 负号：numpy 默认升序，我们要最大在前
    return [(float(scores[i]), int(i)) for i in order]


def main():
    vecs = np.load(INDEX_DIR / "vectors.npy")
    chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
    meta = json.loads((INDEX_DIR / "meta.json").read_text(encoding="utf-8"))

    print(f"索引载入：{meta['num_chunks']} 块 / {meta['dim']} 维 / {meta['model']}")
    print("=" * 62)

    model = SentenceTransformer(meta["model"])

    question = "中国石油2023年的营业收入是多少？"   # 按你的数据改
    q = model.encode(question, normalize_embeddings=True)

    print(f"\n问题：{question}\n")
    for rank, (score, idx) in enumerate(search_numpy(q, vecs), 1):
        text = chunks[idx]["text"][:70].replace("\n", " ")
        print(f"  {rank}. [{score:.4f}] {text}...")

    print("\n" + "=" * 62)
    print("速度对比")

    t0 = time.perf_counter()
    search_python(q, vecs)
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(20):
        search_numpy(q, vecs)
    t_np = (time.perf_counter() - t0) / 20

    print(f"  纯 Python 循环 : {t_py * 1000:9.1f} ms")
    print(f"  numpy 矩阵     : {t_np * 1000:9.2f} ms")
    print(f"  提速           : {t_py / t_np:9.0f} 倍")


if __name__ == "__main__":
    main()