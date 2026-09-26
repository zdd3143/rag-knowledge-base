import numpy as np
from sentence_transformers import SentenceTransformer


docs = [
    "中国石油2023年实现营业收入30110亿元，比上年同期下降7.0%。",
    "中国石化2023年实现营业收入32122亿元，同比下降3.2%。",
    "中国海油2023年油气净产量为6.78亿桶油当量。",
    "水力压裂是利用高压液体在岩层中制造裂缝的技术。",
    "支撑剂的作用是防止压裂后的裂缝重新闭合。",
  ]




model = SentenceTransformer("BAAI/bge-m3")

doc_vecs = model.encode(docs, normalize_embeddings=True)

question = "中国石油2023年的营业收入是多少？"
q_vec = model.encode(question, normalize_embeddings=True)

scores = doc_vecs @ q_vec

for i in np.argsort(-scores)[:3]:
    snippet = docs[i][:80].replace("\n", " ")
    print(f"{scores[i]:.4f}  {docs[i]}")