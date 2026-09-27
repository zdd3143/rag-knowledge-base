"""诊断：正确答案到底在不在索引里？如果在了，排第几？"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_DIR = Path("index")
vecs = np.load(INDEX_DIR / "vectors.npy")
chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))

print(f"索引共 {len(chunks)} 块")
print("=" * 62)

# ---------- 第 1 步：人工找出「答案」在哪一块 ----------
# 先在你自己的年报里搜一下真实数字，把下面的 KEY 改成答案里的特征数字
KEY = "30,110"
found = [i for i, c in enumerate(chunks) if KEY in c["text"]]

print(f"\n【1】含「{KEY}」的块：{found}")
for i in found[:3]:
    print(f"  块 #{i} 内容：{chunks[i]['text'][:180]!r}")

if not found:
    print("  ⚠️ 答案的关键数字压根不在任何块里！")
    print("     可能原因：① KEY 写错了 ② PDF 提取时这个数字没被抽出来")
else:
    # ---------- 第 2 步：看它排第几 ----------
    model = SentenceTransformer("BAAI/bge-m3")
    q = model.encode("中国石油2023年的营业收入是多少？",
                     normalize_embeddings=True)
    scores = vecs @ q
    order = np.argsort(-scores)

    print(f"\n【2】这些块在检索里的排名（共 {len(chunks)} 块）")
    for i in found:
        rank = int(np.where(order == i)[0][0]) + 1
        print(f"  块 #{i}  →  第 {rank} 名，得分 {float(scores[i]):.4f}")

    print(f"\n【3】实际返回的第一名")
    top = order[0]
    print(f"  块 #{top}，得分 {float(scores[top]):.4f}")
    print(f"  {chunks[top]['text'][:180]!r}")
