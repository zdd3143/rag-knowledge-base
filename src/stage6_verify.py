"""核对：模型引用的块里，到底有没有它声称的数字？"""
import json
import re
from pathlib import Path

chunks = json.loads(Path("index/chunks.json").read_text(encoding="utf-8"))
texts = [c["text"] for c in chunks]

# ---- 把模型这次的输出原样粘进来 ----
ANSWER = "中国石油2023年的净利润为1,802.91亿元（按中国企业会计准则计算）[1]，即人民币180,291百万元[5]。"
CITED = [433, 395]          # [1] -> #433, [5] -> #395

print("=" * 72)
print("一、答案里出现的数字")
print("=" * 72)
for n in re.findall(r"[\d,]+\.?\d*", ANSWER):
    if len(n) >= 4:
        print(f"  {n}")

print()
print("=" * 72)
print("二、被引用的块，完整原文 + 关键词核对")
print("=" * 72)
for doc_id in CITED:
    text = texts[doc_id]
    print(f"\n--- 块 #{doc_id}（{len(text)} 字）---")
    print(text)
    print("\n  关键词核对：")
    for key in ["1,802", "180,291", "1802", "净利润", "归属于母公司"]:
        print(f"    「{key}」 {'✓ 出现' if key in text else '✗ 没出现'}")
    print()

print("=" * 72)
print("三、全库搜索：1,802 到底在哪几块里")
print("=" * 72)
hits = [i for i, t in enumerate(texts) if "1,802" in t]
if hits:
    for i in hits:
        print(f"\n  块 #{i}:")
        print(f"    {' '.join(texts[i].split())[:200]}")
else:
    print("  ⚠️ 全库都找不到 1,802 —— 说明这个数字可能是模型编的")