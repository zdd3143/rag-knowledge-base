"""诊断：重排模型到底有没有在工作？"""
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

QUESTION = "中国石油2023年的营业收入是多少？"

pairs = [
    (QUESTION, "本集团实现营业收入人民币30,110.12亿元，比上年同期下降7.0%。"),   # 明确相关
    (QUESTION, "今天天气不错，适合出去打球。"),                                  # 明确无关
    (QUESTION, "505  -  1,285,752  -  -  1,350,257  合计  151,150  336,861"),  # 数字粥
]

scores = reranker.predict(pairs)
for (q, d), s in zip(pairs, scores):
    print(f"  {float(s):.4f}   {d[:45]}")