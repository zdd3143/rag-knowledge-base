"""找出适合做评估题的锚点候选。

评判标准：
  1. 块里含关键指标词（营业收入 / 净利润 / 产量 ...）
  2. 块里有"带千分位且带两位小数"的数字（通常是有意义的具体指标值）
  3. 该数字在全库出现的块数要少（<=5），否则做锚点区分度不够

用法：
    python src/stage7_find_anchors.py
然后把输出里挑出来的锚点，人工写成 data/eval/questions.jsonl
"""
import json
import re
from pathlib import Path

CHUNKS = Path("index/chunks.json")
texts = [c["text"] for c in json.loads(CHUNKS.read_text(encoding="utf-8"))]
print(f"语料共 {len(texts)} 块")

METRICS = [
    "营业收入", "净利润", "归属于母公司", "每股", "原油产量", "天然气产量",
    "平均实现价格", "资产负债率", "经营活动产生的现金流量", "资本性支出",
    "探明储量", "炼油产品", "化工产品", "加油站", "所得税", "销售费用",
]

NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+\.\d{2}")

# 预先算好每个数字出现在多少块里，用于判断区分度
used = set()
total = 0

for kw in METRICS:
    print()
    print("=" * 78)
    print(f"【{kw}】")
    print("=" * 78)
    shown = 0
    for i, t in enumerate(texts):
        if kw not in t:
            continue
        cands = [n for n in NUM_RE.findall(t) if n not in used]
        if not cands:
            continue

        # 挑全库出现次数最少的数字 —— 区分度最高
        best, best_df = None, 10 ** 9
        for n in cands:
            df = sum(1 for x in texts if n in x)
            if df < best_df:
                best, best_df = n, df
        if best is None or best_df > 5:
            continue

        used.add(best)
        total += 1
        flat = " ".join(t.split())
        pos = flat.find(best)
        print(f"\n  锚点 {best}   全库出现 {best_df} 次   块 #{i}")
        print(f"    ...{flat[max(0, pos - 95):pos + 70]}...")

        shown += 1
        if shown >= 3:
            break
    if shown == 0:
        print("  （没有合适候选）")

print()
print("=" * 78)
print(f"共找到 {total} 个可用锚点候选")
print("=" * 78)
print()
print("下一步：从上面挑 25-30 个，每个配一句问句，写进 data/eval/questions.jsonl")
print("格式（每行一个 JSON）：")
print('  {"id": "q001", "category": "factual", "question": "……是多少？", "anchors": ["30,110.12"]}')