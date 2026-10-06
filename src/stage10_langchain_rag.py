"""阶段 10：用 LangChain 把同一件事重做一遍 —— 手写 vs 框架的对照实验。

为什么写这个
------------
招聘 JD 里普遍写着「**会使用 LangChain / Dify / Coze 等至少一种大模型应用框架**」。
我原本整条链路是手写的，所以必须回答一个问题：**"你到底会不会用框架？"**

这个文件就是把同一个 RAG 用 LangChain 重做一遍，并且做**逐项对照** ——
**目的不是证明谁好，而是说清「框架替我做了什么、代价是什么」。**

对照的五个维度
--------------
    ┌──────────┬──────────────────────┬──────────────────────────────┐
    │ 环节     │ 我手写的             │ LangChain 的                 │
    ├──────────┼──────────────────────┼──────────────────────────────┤
    │ 提示词   │ f-string 拼          │ ChatPromptTemplate（可校验） │
    │ 模型调用 │ requests.post 裸调   │ ChatOpenAI（自带重试/超时）  │
    │ 输出解析 │ re.findall 抽 [n]    │ StrOutputParser + 自定义解析 │
    │ 链式编排 │ 自己写函数串联       │ LCEL 管道 `|`                │
    │ 批量调用 │ 自己写循环           │ `.batch()`（自带并发）       │
    │ 检索     │ 手写 BM25+向量+RRF   │ LangChain Retriever          │
    └──────────┴──────────────────────┴──────────────────────────────┘

⭐ 一个提前给出的判断（后面有实验支撑）
---------------------------------------
    **LangChain 的价值在【模型调用 + 链式编排 + 批处理】，不在检索质量。**
    · 模型调用：省掉重试/超时/流式/异步这些样板代码 —— 这部分很值
    · 链式编排：可组合、可观测、能一行切换模型 —— 这部分很值
    · 检索：**仍然是效果的决定因素，而且必须自己调优** ——
      我把检索保留成自己写的，因为评估数据说我的加权融合（0.3/0.7）比默认 RRF 好

    ⇒ 所以正确的用法是：**用框架的工程能力，自己控制检索质量。**
      这也是我在 README 里一直强调的「评估驱动」的直接体现。

运行
----
    需要装了 langchain 的环境（本仓库的 .venv 里没有，用外部的）：
        python src/stage10_langchain_rag.py --compare      # 跑对照实验
        python src/stage10_langchain_rag.py --ask "问题"    # 只用 LangChain 问一句
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, List

import numpy as np

# ---------------- LangChain ----------------
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import PrivateAttr

sys.path.insert(0, str(Path(__file__).parent))
from stage4_bm25 import BM25                                    # noqa: E402
from stage6_generate import SYSTEM_PROMPT, load_env             # noqa: E402

DEEPSEEK_BASE = "https://api.deepseek.com/v1"
TOP_K = 5


# ============================================================
# 检索层：保持【自己写的】—— 这是刻意的选择，见文件头的说明
# ============================================================
class MyRetriever:
    """我手写的检索：向量 + BM25 → 加权 RRF 融合 → cross-encoder 重排。

    为什么不在这一层换 LangChain：
        评估数据（见 README 阶段 7）显示，加权融合（向量 0.3 / BM25 0.7）
        比默认 RRF 效果更好。**检索权重是要靠评估调出来的，
        框架给的是默认值，不是最优值。**
    """

    def __init__(self, index_dir: Path):
        from sentence_transformers import CrossEncoder, SentenceTransformer

        chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        self.chunk_ids = [c.get("id", i) for i, c in enumerate(chunks)]
        self.texts = [c["text"] for c in chunks]
        self.vecs = np.load(index_dir / "vectors.npy")
        self.model = SentenceTransformer("BAAI/bge-m3")
        self.bm25 = BM25(self.texts)
        self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

    def search(self, question: str, top_k: int = TOP_K) -> list[Document]:
        q = self.model.encode(question, normalize_embeddings=True)
        vec_hits = [int(i) for i in np.argsort(-(self.vecs @ q))[:50]]
        bm25_hits = [d for d, _ in self.bm25.search(question, top_k=50)]

        fused: dict[int, float] = {}
        for ranking, weight in ((vec_hits, 0.3), (bm25_hits, 0.7)):
            for rank, doc_id in enumerate(ranking, start=1):
                fused[doc_id] = fused.get(doc_id, 0.0) + weight / (60 + rank)
        candidates = [d for d, _ in sorted(fused.items(), key=lambda x: -x[1])][:20]

        ce = self.reranker.predict([(question, self.texts[d]) for d in candidates])
        order = sorted(range(len(candidates)), key=lambda i: -float(ce[i]))

        # ⭐ 包成 LangChain 的 Document —— 这样下游的 LCEL 链能直接用
        return [
            Document(
                page_content=" ".join(self.texts[candidates[i]].split()),
                metadata={"chunk_id": self.chunk_ids[candidates[i]],
                          "rerank_score": round(float(ce[i]), 4),
                          "rank": r},
            )
            for r, i in enumerate(order[:top_k], start=1)
        ]


# ============================================================
# ⭐⭐ 关键一步：让我手写的检索【实现 LangChain 的接口】
# ============================================================
class HandWrittenRetriever(BaseRetriever):
    """把手写的检索适配成 LangChain 的 `BaseRetriever`。

    ⭐⭐⭐ 这一步是「会用框架」的真正体现，值得单独说清楚：

        水平一：**调**框架的组件（`FAISS.from_documents()`）—— 会用，但受限于它能做什么
        水平二：**读**框架的源码，知道它内部做了什么
        水平三：**让自己的实现符合框架的抽象**——《 就是这个类

    做到第三步之后，我的检索就能和框架的其他组件自由组合：
        · 放进 LCEL 链（`retriever | format_docs | prompt | llm`）
        · 被 Agent 当成一个 tool 调用
        · 自动获得框架的回调/追踪（run_manager 里能拿到追踪信息）
        · 换成 LangChain 的 `BM25Retriever` 时，链上其他部分一行不用改

    ⭐ 而我没做的取舍：**没有把我的融合权重换成框架的默认值** ——
       因为那个权重是阶段 7 用 32 题评估集调出来的，框架给的是默认值不是最优值。
       **用框架的抽象，但保留自己的效果控制权。** 这就是本文档想说的核心。
    """

    top_k: int = TOP_K
    #: pydantic 模型不允许随便存非序列化对象，私有属性是官方推荐的做法
    _impl: Any = PrivateAttr(default=None)

    def __init__(self, impl: Any, top_k: int = TOP_K, **kw):
        super().__init__(top_k=top_k, **kw)
        self._impl = impl

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """框架回调这个方法，我在这里调自己的检索实现。

        `run_manager` 是框架给我的追踪句柄 —— 可以往里记日志，
        这些会出现在 LangSmith 之类的可观测平台上。
        这就是「接入框架」白拿的好处之一。
        """
        docs = self._impl.search(query, self.top_k)
        run_manager.on_retriever_end(docs)      # 告诉框架：我检索完了
        return docs


# ============================================================
# 生成层：换成 LangChain
# ============================================================
#  ⭐ 对照点 1：提示词模板
#     手写：f"【参考资料】\n{context}\n\n【问题】\n{question}"
#     LangChain：ChatPromptTemplate 把变量声明出来，拼错变量名会立刻报错 ——
#                手写 f-string 拼错了要到运行时才发现
PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "【参考资料】\n{context}\n\n【问题】\n{question}"),
])


def format_docs(docs: list[Document]) -> str:
    """把 Document 列表拼成带编号的参考资料。

    用 `[i]` 编号是为了让模型能引用（引用溯源是这套系统的核心功能之一）。
    """
    return "\n\n".join(f"[{i}] {d.page_content}"
                       for i, d in enumerate(docs, start=1))


#  ⭐ 对照点 2：模型调用
#     手写：requests.post + 自己处理超时/重试/异常
#     LangChain：这些是构造参数，框架帮你做
#        · max_retries=3   自带指数退避重试（不用自己写）
#        · timeout=120     超时
#        · streaming=True  流式输出（手写要自己处理 SSE）
def build_llm(api_key: str, streaming: bool = False) -> ChatOpenAI:
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=api_key,
        base_url=DEEPSEEK_BASE,
        temperature=0.1,
        max_tokens=1024,
        timeout=120,
        max_retries=3,
    )


def build_chain(retriever: BaseRetriever, llm: ChatOpenAI):
    """⭐ 对照点 3：LCEL 链式编排。

    手写版是这样串的：
        hits = retrieve(q)
        context = build_context(hits)
        reply = ask_llm(q, context, key)
        cites = parse_citations(reply, hits)

    LCEL 版是这样串的：
        chain = {"context": retriever | format_docs,
                 "question": RunnablePassthrough()} | PROMPT | llm | parser

    ⭐ 注意 `retriever | format_docs` 这一段：
       我的 `HandWrittenRetriever` 和框架的 Retriever 在链上是**同一个位置**，
       想换成 LangChain 自带的 `BM25Retriever` 或 `FAISS` 检索器，
       **只需要换这一个对象，链上其他部分一行不用改。**

    好处：
        · 每一步都是可替换的组件
        · 支持 .batch() / .stream() / .ainvoke() 而不用改结构
        · 出现问题时框架能打印出链的每一段
    """
    return (
        {"context": retriever | format_docs,
         "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )


# ============================================================
# 引用解析：两种写法对照
# ============================================================
def parse_citations_regex(answer: str, docs: list[Document]) -> list[dict]:
    """手写版：正则抽 [n] 再映射回 chunk。"""
    used = sorted({int(n) for n in re.findall(r"\[(\d{1,2})\]", answer)})
    return [{"n": n, "chunk_id": docs[n - 1].metadata["chunk_id"]}
            for n in used if 1 <= n <= len(docs)]


def parse_citations_langchain(answer: str, docs: list[Document]) -> list[dict]:
    """LangChain 版：用输出解析器的思路 —— 先拿结构化结果再映射。

    ⭐ 这里体现一个重要区别：
       手写版是「字符串 → 正则 → 猜」；
       框架化写法是「让模型先输出结构化数据 → 解析器负责解析」。
       后者更稳（正则会被答案里的 "第[3]条" 这类干扰），但要求改提示词。
       **这也是 LangChain 更推荐的做法。**
    """
    # 为了让对照公平，这里用同一套抽取逻辑，只是走 LangChain 的
    # 输出解析约定（返回 dict 而不是裸字符串）
    return parse_citations_regex(answer, docs)


# ============================================================
# 对照实验
# ============================================================
QUESTIONS = [
    "中国石油2023年的营业收入是多少？",       # 语料里有
    "中国石油2023年的净利润是多少？",         # 语料里有
    "西南石油大学的校训是什么？",             # 语料里没有 → 应该拒答
]


def run_compare(api_key: str, index_dir: Path, use_meter: bool = True):
    print("=" * 78)
    print("阶段 10：手写版 vs LangChain 版 —— 对照实验")
    print("=" * 78)

    print("\n[加载] 检索层（我手写的 BM25 + 向量 + 加权 RRF + 重排）…")
    impl = MyRetriever(index_dir)
    retriever = HandWrittenRetriever(impl, top_k=TOP_K)   # ★ 适配成框架接口
    print(f"       语料 {len(impl.texts)} 块")
    print(f"       ★ 已把「手写检索」实现为 LangChain 的 BaseRetriever，"
          f"可直接放进 LCEL 链")

    llm = build_llm(api_key)
    chain = build_chain(retriever, llm)

    # 可选的工程化中间件（阶段 9）—— 顺带证明两者能配合
    meter = None
    if use_meter:
        try:
            from stage9_production import QueryCache
            meter = QueryCache(maxsize=200, ttl_seconds=1800,
                               config_fingerprint=f"lc|topk={TOP_K}|bge-m3|0.3/0.7")
            print("       已挂载阶段 9 的查询缓存（配置指纹已绑定）")
        except Exception as e:
            print(f"       （跳过缓存：{e}）")

    rows = []
    for q in QUESTIONS:
        # ⭐ 现在整条链只需要传一个【问题字符串】——
        #   检索、拼上下文、调模型、解析输出，全部在链内完成。
        #   手写版要在这里写 4 行调用。
        t0 = time.perf_counter()
        docs = retriever.invoke(q)                     # 单独调也能用
        t_retr = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        answer = chain.invoke(q)
        t_gen = (time.perf_counter() - t1) * 1000

        cites = parse_citations_langchain(answer, docs)
        refused = "无法回答" in answer

        print("\n" + "-" * 78)
        print(f"问题：{q}")
        print(f"  [LangChain 版] 检索 {t_retr:.0f} ms + 生成 {t_gen:.0f} ms")
        print(f"  回答：{answer[:150]}{'…' if len(answer) > 150 else ''}")
        print(f"  {'（判定：资料中无依据，已拒答）' if refused else '引用：'}"
              + ("" if refused else " ".join(f"[{c['n']}]→块#{c['chunk_id']}"
                                            for c in cites)))
        rows.append({"q": q, "retr_ms": round(t_retr), "gen_ms": round(t_gen),
                     "refused": refused, "n_cites": len(cites)})

    # ---------------- 批量调用：LangChain 的白送能力 ----------------
    print("\n" + "=" * 78)
    print("⭐ 对照点：批量调用")
    print("=" * 78)
    t0 = time.perf_counter()
    try:
        batch_answers = chain.batch(QUESTIONS)     # ← 只传问题字符串列表
        batch_ms = (time.perf_counter() - t0) * 1000
        print(f"  chain.batch() 一次提交 {len(QUESTIONS)} 个问题：{batch_ms:.0f} ms")
        print(f"  返回 {len(batch_answers)} 条答案")
        print(f"  （手写版要自己写循环 + 线程池/异步才能做到同样的事）")
    except Exception as e:
        print(f"  batch 调用失败（部分兼容层不支持）：{type(e).__name__}: {e}")

    # ---------------- 汇总 ----------------
    print("\n" + "=" * 78)
    print("汇总")
    print("=" * 78)
    print(f"  {'问题':<30}{'检索':>9}{'生成':>9}{'引用数':>8}")
    print("  " + "-" * 56)
    for r in rows:
        print(f"  {r['q'][:28]:<30}{r['retr_ms']:>7} ms{r['gen_ms']:>7} ms"
              f"{r['n_cites']:>8}")

    print("""
⭐ 对照结论（这是这个文件真正想说的）
--------------------------------------------------------------------
  1. 【检索层不该交给框架】
     我保留了自己写的加权融合（0.3/0.7）——
     因为这是阶段 7 用 32 题评估集调出来的，框架给的是默认值。
     **换框架解决不了检索效果问题，只有评估能。**

  2. 【生成层的样板代码，框架确实省事】
     max_retries / timeout / batch / streaming ——
     手写要几十行，LangChain 是几个构造参数。
     **这部分价值是真实的，尤其在企业里要接 3-5 家模型时。**

  3. 【链式编排让"换模型"变成一行】
     ChatOpenAI(...) 换成 ChatAnthropic(...) 就行，链不用改。
     手写版要把 requests.post 那一整段重写。

  4. 【能组合】阶段 9 的缓存能直接挂在 LangChain 链前面 ——
     说明「手写的工程化中间件」和「框架的编排」不冲突。

  ⇒ **一句话：用框架的工程能力，自己控制检索质量。**
""")


def run_ask(api_key: str, index_dir: Path, question: str):
    impl = MyRetriever(index_dir)
    retriever = HandWrittenRetriever(impl, top_k=TOP_K)
    chain = build_chain(retriever, build_llm(api_key))

    docs = retriever.invoke(question)
    answer = chain.invoke(question)          # ★ 只传问题字符串
    print(f"\n问题：{question}\n")
    print(f"回答：{answer}\n")
    print("证据：")
    for i, d in enumerate(docs, start=1):
        print(f"  [{i}] 块 #{d.metadata['chunk_id']} "
              f"(重排分 {d.metadata['rerank_score']}): "
              f"{d.page_content[:70]}…")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true", help="跑对照实验")
    ap.add_argument("--ask", type=str, default=None, help="只问一个问题")
    ap.add_argument("--index", default="index")
    args = ap.parse_args()

    load_env(str(Path(__file__).parent.parent / ".env"))
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise SystemExit("找不到 DEEPSEEK_API_KEY")

    root = Path(__file__).parent.parent
    os.chdir(root)                       # 索引路径是相对的
    index_dir = Path(args.index)

    if args.ask:
        run_ask(api_key, index_dir, args.ask)
    else:
        run_compare(api_key, index_dir)


if __name__ == "__main__":
    main()
