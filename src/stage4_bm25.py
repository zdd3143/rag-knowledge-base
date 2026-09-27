"""BM25：基于词频与逆文档频率的关键词检索。
"""
import math
from collections import Counter, defaultdict

import jieba

 # 加领域词 
DOMAIN_WORDS = [
        "中国石油", "中国石化", "中国海油", "中海油服", "海油工程",
        "营业收入", "营业总收入", "净利润", "归母净利润", "利润总额",
        "资产负债率", "毛利率", "经营活动现金流量净额", "每股收益",
        "原油产量", "天然气产量", "油气当量", "探明储量", "资本开支",
    ]
for _word in DOMAIN_WORDS:
    jieba.add_word(_word)
   

class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        """
        k1: 词频饱和系数。词频再高，贡献也有上限（默认 1.5）
        b:  长度归一化强度。0=完全不归一化，1=完全归一化（默认 0.75）
        """
        self.k1 = k1
        self.b = b

        # 1) 分词
        self.docs = [list(jieba.cut(text)) for text in corpus]
        self.N = len(self.docs)
        self.avgdl = sum(len(d) for d in self.docs) / self.N

        # 2) 倒排索引：词 -> {文档编号: 该词出现次数}
        self.inverted = defaultdict(dict)
        for i, doc in enumerate(self.docs):
            for term, freq in Counter(doc).items():
                self.inverted[term][i] = freq

        # 3) 每个词的 IDF
        #    公式 log(1 + (N - df + 0.5)/(df + 0.5))
        #    df = 含有这个词的文档数。df 越小（越稀有）→ IDF 越大 → 越有区分度
        self.idf = {}
        for term, postings in self.inverted.items():
            df = len(postings)
            self.idf[term] = math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def search(self, query, top_k=5):
        terms = [t for t in jieba.cut(query) if t.strip()]
        scores = defaultdict(float)

        for term in terms:
            if term not in self.inverted:
                continue                       # 词表里没有，跳过
            idf = self.idf[term]
            for doc_id, freq in self.inverted[term].items():
                dl = len(self.docs[doc_id])
                # BM25 的核心公式
                denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc_id] += idf * freq * (self.k1 + 1) / denom

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return ranked


if __name__ == "__main__":
    import json
    from pathlib import Path

    chunks = json.loads(Path("index/chunks.json").read_text(encoding="utf-8"))
    texts = [c["text"] for c in chunks]

   
    #建索引
    print(f"建立 BM25 索引：{len(texts)} 块")
    bm25 = BM25(texts)
    print(f"词表大小：{len(bm25.inverted):,} 个不同的词")

    #IDF 诊断 
    print("\n【诊断】这些词的区分度到底差多少？")
    print(f"{'词':<12}{'出现在多少块里':>14}{'IDF':>10}")
    for term in ["的", "年", "营业", "收入", "营业收入", "中国", "石油", "中国石油"]:
        if term in bm25.inverted:
            df = len(bm25.inverted[term])
            print(f"{term:<12}{df:>14}{bm25.idf[term]:>10.3f}")
        else:
            print(f"{term:<12}{'不在词表里':>14}")

    #检索
    question = "中国石油2023年的营业收入是多少？"
    print("=" * 62)
    print(f"\n问题：{question}")
    print(f"分词结果：{list(jieba.cut(question))}\n")

    for rank, (doc_id, score) in enumerate(bm25.search(question, top_k=5), 1):
        text = texts[doc_id][:70].replace("\n", " ")
        mark = "  ← 正确答案！" if doc_id == 44 else ""
        print(f"  {rank}. [{score:.3f}] 块 #{doc_id}: {text}...{mark}")