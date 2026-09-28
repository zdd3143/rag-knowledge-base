"""阶段 6：接入大模型，生成带引用的答案。


"""
import json
import os
import re
from pathlib import Path

import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from stage4_bm25 import BM25

TOP_K = 5

SYSTEM_PROMPT = """你是一个严谨的知识库问答助手。

必须遵守三条规则：
1. 只根据【参考资料】回答，不要使用你自己的知识，也不要推测。
2. 每个结论后面标注来源编号，格式为 [1]、[2]，可同时标注多个，如 [1][3]。
3. 如果参考资料里没有答案，直接回答「根据现有资料无法回答该问题」，不要编造。

回答要简洁，直接给结论。"""


def load_env(path=".env"):
    
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


# ---------------------------------------------------------------- 检索

def build_retriever():
    chunks = json.loads(Path("index/chunks.json").read_text(encoding="utf-8"))
    texts = [c["text"] for c in chunks]
    vecs = np.load("index/vectors.npy")

    model = SentenceTransformer("BAAI/bge-m3")
    bm25 = BM25(texts)
    return texts, vecs, model, bm25


def retrieve(question, texts, vecs, model, bm25, top_k=TOP_K):
    """两路检索 + RRF 融合，返回前 top_k 个块编号"""
    q = model.encode(question, normalize_embeddings=True)
    scores = vecs @ q
    vec_hits = [(int(i), float(scores[i])) for i in np.argsort(-scores)[:50]]
    bm25_hits = bm25.search(question, top_k=50)

    fused = {}
    for ranking in (vec_hits, bm25_hits):
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (60 + rank)

    return sorted(fused.items(), key=lambda x: -x[1])[:top_k]


# ---------------------------------------------------------------- 生成

def build_context(hits, texts):
    """把检索到的块拼成带编号的参考资料"""
    lines = []
    for i, (doc_id, _score) in enumerate(hits, 1):
        text = " ".join(texts[doc_id].split())      # 把换行压成空格，省 token
        lines.append(f"[{i}] {text}")
    return "\n\n".join(lines)


def ask_llm(question, context, api_key):
    """调用 DeepSeek 官方 API"""
    user_prompt = f"【参考资料】\n{context}\n\n【问题】\n{question}"

    resp = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,       # 低温度：问答要稳定，不要发挥
            "max_tokens": 1024,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def parse_citations(answer, hits):
    """从答案里抽出 [n]，映射回具体的块"""
    used = sorted({int(n) for n in re.findall(r"\[(\d{1,2})\]", answer)})
    return [(n, hits[n - 1][0]) for n in used if 1 <= n <= len(hits)]


# ---------------------------------------------------------------- 主流程

def answer_one(question, texts, vecs, model, bm25, api_key, top_k=TOP_K):
    hits = retrieve(question, texts, vecs, model, bm25, top_k=top_k)
    context = build_context(hits, texts)
    reply = ask_llm(question, context, api_key)
    return reply, hits, parse_citations(reply, hits)


def main():
    load_env()
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise SystemExit("找不到 DEEPSEEK_API_KEY，请在 .env 里配置")

    print("加载索引与模型…")
    texts, vecs, model, bm25 = build_retriever()
    print(f"语料 {len(texts)} 块，就绪\n")

    questions = [
        "中国石油2023年的营业收入是多少？",        # 语料里有
        "中国石油2023年的净利润是多少？",          # 语料里有
        "西南石油大学的校训是什么？",              # 语料里没有 → 应该拒答
        "量子纠缠的贝尔不等式怎么验证？",          # 完全无关 → 应该拒答
    ]

    for q in questions:
        print("=" * 72)
        print(f"问题：{q}")
        print("-" * 72)
        reply, hits, cites = answer_one(q, texts, vecs, model, bm25, api_key)

        print(f"回答：{reply}")

        refused = "无法回答" in reply
        print(f"\n{'（系统判定：知识库中无依据，已拒答）' if refused else '引用来源：'}")
        if not refused:
            for n, doc_id in cites:
                snippet = " ".join(texts[doc_id].split())[:60]
                print(f"  [{n}] 块 #{doc_id}: {snippet}...")
        elif not cites:
            pass
        print()


if __name__ == "__main__":
    main()