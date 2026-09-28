"""阶段 7：评估检索质量。"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from stage4_bm25 import BM25

K_VALUES = (1, 3, 5, 10)
EVAL_PATH = "eval/questions.jsonl"


# ------------------------------------------------------------ 评估集

def load_items(path=EVAL_PATH):
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            items.append(json.loads(line))
    return items


def resolve_gold(items, texts):
    """把锚点（原文片段）解析成 chunk 编号。"""
    out = []
    for it in items:
        gold, missing = set(), []
        for anchor in it.get("anchors", []):
            hit = [i for i, t in enumerate(texts) if anchor in t]
            if hit:
                gold.update(hit)
            else:
                missing.append(anchor)
        out.append({**it, "gold": sorted(gold), "missing": missing})
    return out


# ------------------------------------------------------------ 指标

def hit_at_k(ranked, gold, k):
    """前 k 个里有没有正确答案"""
    return float(any(g in ranked[:k] for g in gold))


def mrr_of(ranked, gold):
    """正确答案排名的倒数"""
    for i, d in enumerate(ranked, start=1):
        if d in gold:
            return 1.0 / i
    return 0.0


# ------------------------------------------------------------ 检索配置

def make_retrievers(texts, vecs, model, bm25, reranker):
    """返回 {配置名: 检索函数}，函数接收问题，返回按相关性排序的 chunk 编号"""

    def vector(question, top_n=50):
        q = model.encode(question, normalize_embeddings=True)
        scores = vecs @ q
        return [int(i) for i in np.argsort(-scores)[:top_n]]

    def bm25_only(question, top_n=50):
        return [d for d, _ in bm25.search(question, top_k=top_n)]

    def fuse_weighted(question, w_vec, w_bm25, top_n=50):
        """加权 RRF：每一路的贡献乘以它的权重。

        为什么需要权重：等权（0.5/0.5）在「两路水平相当」时是合理的，
        但一旦一路明显更强，等权就是在**稀释强信号** ——
        弱的那一路会把强的那一路排上来的正确答案挤下去。

        实测（24 条问题）：
            纯 BM25   Hit@5 0.8333
            纯向量    Hit@5 0.2917
            RRF 等权  Hit@5 0.5417   <- 比最强的单路还差！
        """
        fused = {}
        for ranking, weight in ((vector(question), w_vec),
                                (bm25_only(question), w_bm25)):
            for rank, d in enumerate(ranking, start=1):
                fused[d] = fused.get(d, 0.0) + weight / (60 + rank)
        return [d for d, _ in sorted(fused.items(), key=lambda x: -x[1])][:top_n]

    def rerank(question, candidates, head_n=20):
        """对前 head_n 个候选做 cross-encoder 精排"""
        head = candidates[:head_n]
        scores = reranker.predict([(question, texts[d]) for d in head])
        order = sorted(range(len(head)), key=lambda i: -float(scores[i]))
        return [head[i] for i in order] + candidates[head_n:]

    return {
        "纯向量":             lambda q: vector(q),
        "纯 BM25":            lambda q: bm25_only(q),
        "RRF 等权(0.5/0.5)":  lambda q: fuse_weighted(q, 0.5, 0.5),
        "加权(0.4/0.6)":      lambda q: fuse_weighted(q, 0.4, 0.6),
        "加权(0.3/0.7)":      lambda q: fuse_weighted(q, 0.3, 0.7),
        "加权(0.2/0.8)":      lambda q: fuse_weighted(q, 0.2, 0.8),
        "加权(0.3/0.7)+重排":  lambda q: rerank(q, fuse_weighted(q, 0.3, 0.7)),
    }


# ------------------------------------------------------------ 主流程

def main():
    texts = [
        c["text"]
        for c in json.loads(Path("index/chunks.json").read_text(encoding="utf-8"))
    ]
    vecs = np.load("index/vectors.npy")

    items = resolve_gold(load_items(), texts)

    print(f"评估集共 {len(items)} 条")
    broken = [it for it in items if it["missing"]]
    for it in broken:
        print(f"  [!] {it['id']} 锚点找不到：{it['missing']}")
    if broken:
        print(f"  -> {len(broken)} 条有问题，请检查锚点\n")

    scored = [it for it in items if it["category"] != "unanswerable" and it["gold"]]
    print(f"参与检索评估：{len(scored)} 条（排除不可答 + 无锚点）\n")

    if not scored:
        raise SystemExit("没有可评估的问题，请先补充 eval/questions.jsonl")

    print("加载模型…")
    model = SentenceTransformer("BAAI/bge-m3")
    bm25 = BM25(texts)
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
    retrievers = make_retrievers(texts, vecs, model, bm25, reranker)

    print()
    print("=" * 74)
    header = f"{'配置':<14}" + "".join(f"{'Hit@' + str(k):>10}" for k in K_VALUES) + f"{'MRR':>10}"
    print(header)
    print("-" * 74)

    results = {}
    by_cat = {}
    for name, fn in retrievers.items():
        per_k = {k: [] for k in K_VALUES}
        mrrs = []
        cat_h5 = defaultdict(list)
        cat_mrr = defaultdict(list)

        for it in scored:
            ranked = fn(it["question"])
            h5 = hit_at_k(ranked, it["gold"], 5)
            m = mrr_of(ranked, it["gold"])
            for k in K_VALUES:
                per_k[k].append(hit_at_k(ranked, it["gold"], k))
            mrrs.append(m)
            cat_h5[it["category"]].append(h5)
            cat_mrr[it["category"]].append(m)

        row = f"{name:<20}"
        for k in K_VALUES:
            row += f"{np.mean(per_k[k]):>10.4f}"
        row += f"{np.mean(mrrs):>10.4f}"
        print(row)

        results[name] = {
            f"hit@{k}": round(float(np.mean(per_k[k])), 4) for k in K_VALUES
        } | {"mrr": round(float(np.mean(mrrs)), 4)}

        by_cat[name] = {
            c: {
                "n": len(cat_h5[c]),
                "hit@5": round(float(np.mean(cat_h5[c])), 4),
                "mrr": round(float(np.mean(cat_mrr[c])), 4),
            }
            for c in sorted(cat_h5)
        }

    print("=" * 80)

    # ---- 按问题类型细分：看每种方法各自擅长什么 ----
    cats = sorted({c for v in by_cat.values() for c in v})
    first = by_cat[next(iter(by_cat))]
    labels = {c: f"{c}(n={first[c]['n']})" for c in cats}

    print("\n按问题类型细分（Hit@5 / MRR）")
    print("-" * 80)
    print(f"{'配置':<20}" + "".join(f"{labels[c]:>22}" for c in cats))
    print("-" * 80)
    for name in retrievers:
        row = f"{name:<20}"
        for c in cats:
            cell = by_cat[name][c]
            row += f"{cell['hit@5']:.3f} / {cell['mrr']:.3f}".rjust(22)
        print(row)
    print("=" * 80)

    # 落盘，方便对比实验
    out = Path("outputs")
    out.mkdir(exist_ok=True)
    (out / "eval_results.json").write_text(
        json.dumps({"overall": results, "by_category": by_cat},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n结果已保存到 outputs/eval_results.json")


if __name__ == "__main__":
    main()