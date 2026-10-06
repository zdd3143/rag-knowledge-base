"""阶段 8：把检索 + 生成包成 HTTP 服务。"""
import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from sentence_transformers import CrossEncoder, SentenceTransformer

from stage4_bm25 import BM25
from stage6_generate import SYSTEM_PROMPT, load_env
from stage9_production import MeteredLLM, QueryCache, RetryPolicy

#: 检索配置指纹 —— 一旦改动检索逻辑，务必同步改这里。
#: 它会被拼进缓存 key，**保证"换了检索配置之后旧缓存自动失效"**。
#: 忘了改的后果是：返回旧配置下的答案，而且完全看不出来（静默错误）。
RETRIEVAL_FINGERPRINT = "topk=5|bge-m3|rrf:vec0.3+bm25_0.7|rerank:v2-m3"

# 全局状态：服务启动时填充，所有请求共享
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型 —— 只加载一次。

    这是整个服务化里最关键的一处设计。
    如果每次请求都重新加载，单次响应要 40-60 秒，服务等于不可用。
    """
    load_env()
    index = Path("index")
    chunks = json.loads((index / "chunks.json").read_text(encoding="utf-8"))

    print(f"[启动] 语料 {len(chunks)} 块，开始加载模型（约 40-60 秒）…")
    STATE["texts"] = [c["text"] for c in chunks]
    STATE["vecs"] = np.load(index / "vectors.npy")
    STATE["model"] = SentenceTransformer("BAAI/bge-m3")
    STATE["bm25"] = BM25(STATE["texts"])
    STATE["reranker"] = CrossEncoder("BAAI/bge-reranker-v2-m3")
    STATE["api_key"] = os.environ.get("DEEPSEEK_API_KEY", "")

    # ---------- 生产化的三件事（阶段 9）----------
    #  ① 成本可观测  ② 查询缓存  ③ 重试与降级
    #  它们做成可插拔的中间件，检索逻辑一行没动。
    #
    #  ⭐ 缓存 key 里绑定了【检索配置指纹】：
    #     改了权重/模型/top_k 而忘记清缓存 → 指纹变了 → 缓存自动失效。
    #     这比"记得手动清"可靠 —— 把可能的静默错误变成不会发生。
    STATE["cache"] = QueryCache(maxsize=500, ttl_seconds=3600,
                                config_fingerprint=RETRIEVAL_FINGERPRINT)
    STATE["llm"] = MeteredLLM(
        STATE["api_key"],
        cache=STATE["cache"],
        policy=RetryPolicy(max_attempts=3, base_delay=1.0),
    ) if STATE["api_key"] else None

    print(f"[启动] 服务就绪（缓存指纹 {RETRIEVAL_FINGERPRINT[:24]}…）")

    yield

    STATE.clear()
    print("[关闭] 已释放")


class UTF8JSONResponse(JSONResponse):
    """让 JSON 响应显式声明 charset=utf-8。

    为什么需要：JSON 按 RFC 8259 规定就是 UTF-8，所以 FastAPI 默认只发
    `application/json`、不带 charset。现代浏览器能正确处理，
    但 **PowerShell 的 Invoke-RestMethod 等客户端会按 Latin-1 解码**，
    中文就变成 `åæ¬éå¢` 这种乱码 —— 会让人误以为是接口 bug。

    显式声明后，这类客户端也能正确解码。
    这是个很小的改动，但能省掉一整轮"到底是接口坏了还是终端坏了"的排查。
    """

    media_type = "application/json; charset=utf-8"


app = FastAPI(
    title="从零实现的 RAG 检索服务",
    lifespan=lifespan,
    default_response_class=UTF8JSONResponse,
)


# ---------------------------------------------------------------- 数据模型

class Query(BaseModel):
    question: str
    top_k: int = 5


# ---------------------------------------------------------------- 检索

def retrieve(question: str, top_k: int):
    """加权融合 + 重排 —— 参数取自阶段 7 的评估结论（加权 0.3/0.7）。"""
    texts = STATE["texts"]

    q = STATE["model"].encode(question, normalize_embeddings=True)
    vec_scores = STATE["vecs"] @ q
    vec_hits = [int(i) for i in np.argsort(-vec_scores)[:50]]
    bm25_hits = [d for d, _ in STATE["bm25"].search(question, top_k=50)]

    # 加权 RRF：0.3 给向量，0.7 给 BM25（评估出来的权重）
    fused: dict[int, float] = {}
    for ranking, weight in ((vec_hits, 0.3), (bm25_hits, 0.7)):
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (60 + rank)

    candidates = [d for d, _ in sorted(fused.items(), key=lambda x: -x[1])][:20]

    # cross-encoder 精排
    ce = STATE["reranker"].predict([(question, texts[d]) for d in candidates])
    order = sorted(range(len(candidates)), key=lambda i: -float(ce[i]))
    return [(candidates[i], float(ce[i])) for i in order[:top_k]]


def _ensure_ready():
    if "texts" not in STATE:
        raise HTTPException(503, "服务正在加载模型，请稍后重试")


# ---------------------------------------------------------------- 接口

@app.get("/health")
def health():
    """健康检查：模型是否加载好、索引里有多少块。

    为什么需要它：模型加载要 40-60 秒，这期间服务还不能用。
    客户端应该先轮询 /health，通过了再发正式请求。
    """
    ready = "texts" in STATE
    return {
        "status": "ok" if ready else "loading",
        "chunks": len(STATE.get("texts", [])),
        "has_api_key": bool(STATE.get("api_key")),
    }


@app.post("/search")
def search(q: Query):
    """只做检索，不调用大模型 —— 快、免费。"""
    _ensure_ready()
    hits = retrieve(q.question, q.top_k)
    return {
        "question": q.question,
        "hits": [
            {
                "rank": i,
                "chunk_id": d,
                "score": round(s, 4),
                "text": STATE["texts"][d],
            }
            for i, (d, s) in enumerate(hits, start=1)
        ],
    }


@app.post("/ask")
def ask(q: Query):
    """完整问答：检索 + 生成 + 引用溯源。

    ⭐ 相比最初的版本，这里换成了阶段 9 的 MeteredLLM ——
    一次调用就同时获得：缓存、重试降级、token 计量。
    接口签名和返回结构基本没变，**说明中间件是真正的可插拔**。
    """
    _ensure_ready()
    llm: MeteredLLM | None = STATE.get("llm")
    if llm is None:
        raise HTTPException(500, "未配置 DEEPSEEK_API_KEY")

    hits = retrieve(q.question, q.top_k)
    context = "\n\n".join(
        f"[{i}] {' '.join(STATE['texts'][d].split())}"
        for i, (d, _) in enumerate(hits, start=1)
    )
    user_prompt = f"【参考资料】\n{context}\n\n【问题】\n{q.question}"

    r = llm.ask(q.question, SYSTEM_PROMPT, user_prompt)
    answer = r["answer"]

    used = sorted({int(n) for n in re.findall(r"\[(\d{1,2})\]", answer)})
    return {
        "answer": answer,
        "refused": "无法回答" in answer,
        # ---- 工程化元信息：一次问答花了多少钱、有没有走缓存、重试过几次 ----
        "meta": {
            "from_cache": r["from_cache"],
            "degraded": r.get("degraded", False),
            "cost_yuan": r["cost_yuan"],
            "prompt_tokens": r.get("prompt_tokens", 0),
            "completion_tokens": r.get("completion_tokens", 0),
            "latency_ms": r.get("latency_ms", 0),
            "attempts": r.get("attempts", 1),
        },
        "retrieved": [
            {"rank": i, "chunk_id": d, "score": round(s, 4),
             "text": STATE["texts"][d]}
            for i, (d, s) in enumerate(hits, start=1)
        ],
        "citations": [
            {"n": n, "chunk_id": hits[n - 1][0]}
            for n in used if 1 <= n <= len(hits)
        ],
    }


# ---------------------------------------------------------------- 可观测

@app.get("/metrics")
def metrics():
    """成本与健康度指标。

    ⭐ 为什么这个接口重要：
        企业里 RAG 服务上线后，最常被问的三个问题是
            「一个月花多少钱？」「缓存有用吗？」「模型服务稳不稳？」
        没有这个接口，就只能靠猜。

    ⚠️ 注意 `saved_yuan_estimate` 是**估算值** ——
       用「已发生的平均单次成本 × 命中次数」推的。
       被缓存的问题本来会花多少，无法反事实测量。
       对外汇报时要说清这是 estimate。
    """
    llm: MeteredLLM | None = STATE.get("llm")
    cache: QueryCache | None = STATE.get("cache")
    if llm is None:
        return {"status": "no_api_key", "cache": {
            "size": cache.size if cache else 0}}
    m = llm.metrics()
    m["status"] = "ok"
    m["retrieval_fingerprint"] = RETRIEVAL_FINGERPRINT
    return m


@app.post("/cache/clear")
def cache_clear():
    """手动清缓存 —— 改了提示词但要保留检索配置时用得上。"""
    cache: QueryCache | None = STATE.get("cache")
    if cache is None:
        raise HTTPException(503, "缓存未初始化")
    cache.clear()
    return {"ok": True, "size": cache.size}


# ---------------------------------------------------------------- 演示页面
#
#   /        给人看的演示页（就是这个）
#   /docs    FastAPI 自动生成的 Swagger 调试台，是给开发者测接口用的，
#            上面那一堆 Try it out / Execute / Schema 不是给用户看的，不用管它

#     r""" 前面的 r 表示 raw string：里面的反斜杠不再被 Python 当转义符。
#     必须这么做 —— 下面的 JS 里有正则 /\s+/g，如果用普通字符串，
#     Python 会把 \s 当"无效转义"告警；而要是写成 \n，Python 会真的
#     把它替换成换行符，把 JS 代码写坏。这类 bug 极难排查。
DEMO_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>从零实现的 RAG 检索系统</title>
<style>
  :root { --blue:#2563eb; --bg:#f6f7f9; --line:#e4e7eb; --muted:#6b7280; }
  * { box-sizing:border-box; }
  body { font-family:system-ui,"Microsoft YaHei","PingFang SC",sans-serif;
         background:var(--bg); color:#1f2328; line-height:1.65;
         max-width:900px; margin:0 auto; padding:32px 20px 80px; }
  h1 { font-size:22px; margin:0 0 6px; }
  .sub { color:var(--muted); font-size:13.5px; }
  .pill { display:inline-block; padding:2px 10px; border-radius:999px;
          font-size:12px; margin-left:8px; vertical-align:2px; font-weight:400; }
  .pill.ok { background:#dcfce7; color:#166534; }
  .pill.loading { background:#fef3c7; color:#92400e; }

  .step { font-size:13px; color:var(--muted); margin:22px 0 8px; }
  .step b { color:var(--blue); }

  .searchbar { display:flex; gap:8px; }
  input[type=text] { flex:1; padding:12px 14px; font-size:15px;
        border:1px solid #cbd0d6; border-radius:8px; outline:none;
        font-family:inherit; background:#fff; }
  input[type=text]:focus { border-color:var(--blue);
        box-shadow:0 0 0 3px rgba(37,99,235,.12); }
  button { padding:12px 20px; font-size:15px; font-family:inherit;
        border:0; border-radius:8px; cursor:pointer;
        background:var(--blue); color:#fff; white-space:nowrap; }
  button:hover:not(:disabled) { background:#1d4ed8; }
  button:disabled { opacity:.5; cursor:not-allowed; }
  button.ghost { background:#fff; color:#374151; border:1px solid #cbd0d6;
        padding:7px 13px; font-size:13px; }
  button.ghost:hover:not(:disabled) { background:#f3f4f6; }

  .examples { margin-top:14px; font-size:13px; color:var(--muted); }
  .chip { display:inline-block; margin:6px 6px 0 0; padding:5px 11px;
        background:#fff; border:1px solid var(--line); border-radius:999px;
        cursor:pointer; font-size:12.5px; color:#374151; }
  .chip:hover { border-color:var(--blue); color:var(--blue); }

  .hint { margin-top:14px; padding:12px 15px; background:#fff;
        border:1px solid var(--line); border-radius:8px;
        font-size:12.5px; color:var(--muted); }
  .hint b { color:#374151; }

  .card { background:#fff; border:1px solid var(--line); border-radius:10px;
        padding:16px 18px; margin-top:16px; }
  .card > h2 { font-size:14px; margin:0 0 10px; color:#374151;
        display:flex; align-items:baseline; gap:8px; flex-wrap:wrap; }
  .card > h2 .tag { font-weight:400; font-size:12px; color:var(--muted); }

  .answer { white-space:pre-wrap; font-size:15.5px; }
  .card.refused { background:#fffbeb; border-color:#fde68a; }

  .hit { padding:12px 0; border-top:1px dashed #e9ecef; }
  .hit:first-of-type { border-top:0; padding-top:4px; }
  .hit .meta { font-size:12.5px; color:var(--muted); margin-bottom:5px; }
  .rank { display:inline-block; width:20px; height:20px; line-height:20px;
        text-align:center; background:var(--blue); color:#fff;
        border-radius:5px; font-size:11.5px; margin-right:7px; }
  .hit .txt { font-size:14px; color:#374151; white-space:normal;
        max-height:5.2em; overflow:hidden; position:relative; }
  /* 折叠时底部渐变淡出，暗示「下面还有内容」 */
  .hit .txt::after { content:""; position:absolute; left:0; right:0; bottom:0;
        height:1.8em; background:linear-gradient(rgba(255,255,255,0),
        #fff 70%); pointer-events:none; }
  .hit .txt.open { max-height:none; }
  .hit .txt.open::after { display:none; }
  .toggle { margin-top:8px; padding:5px 12px; font-size:12.5px;
        font-family:inherit; background:#fff; color:var(--blue);
        border:1px solid #dbe3f0; border-radius:6px; cursor:pointer; }
  .toggle:hover { background:#f2f6ff; }

  .status { margin-top:18px; font-size:13.5px; color:var(--muted); }
  .spinner { display:inline-block; width:13px; height:13px; margin-right:8px;
        border:2px solid #d1d5db; border-top-color:var(--blue);
        border-radius:50%; animation:spin .7s linear infinite;
        vertical-align:-2px; }
  @keyframes spin { to { transform:rotate(360deg); } }
  .card.err { background:#fef2f2; border-color:#fecaca; color:#991b1b; }
</style>
</head>
<body>

<h1>从零实现的 RAG 检索系统 <span id="pill" class="pill loading">检测中…</span></h1>
<div class="sub">
  语料：中国石油年报 · <span id="chunks">–</span> 个 chunk
  &nbsp;|&nbsp; BM25 + 向量检索 → 加权融合 → cross-encoder 重排 → 大模型生成
</div>

<div class="step"><b>第 ① 步</b> —— 在下面的框里输入你的问题</div>
<div class="searchbar">
  <input id="q" type="text" autocomplete="off"
         placeholder="在这里输入问题，例如：中国石油2023年的营业收入是多少？">
  <button id="bask" onclick="run('ask')">提问</button>
</div>

<div class="examples">
  不知道问什么？点一个试试 →
  <div id="ex"></div>
</div>

<div class="hint">
  <b>两个按钮的区别：</b>
  「<b>提问</b>」= 检索 + 让大模型写答案（会调用 DeepSeek API，慢一点）；
  「<b>只看检索</b>」= 只找出相关段落、不调用大模型（快、免费）。
  <br><br>
  <b>页面下方会出现什么：</b>
  上面是模型给出的<b>回答</b>（方括号里的数字是引用编号）；
  下面是它实际读到的<b>证据段落</b>——回答里的 [1] 就对应证据区的第 1 条。
  这样你能看出答案是<b>基于哪些原文</b>说出来的，而不是凭空生成的。
</div>

<div class="step"><b>第 ② 步</b> —— 点「提问」，然后往下看结果
  <button class="ghost" onclick="run('search')">只看检索（快、免费）</button>
</div>

<div id="out"></div>

<script>
var EXAMPLES = [
  "中国石油2023年的营业收入是多少？",
  "中国石油2023年归属于母公司股东的净利润是多少？",
  "中国石油2023年的资本性支出是多少？",
  "中国石油2023年的营收规模有多大？",
  "西南石油大学的校训是什么？"
];

document.getElementById("ex").innerHTML = EXAMPLES.map(function (t) {
  return '<span class="chip" onclick="fill(this)">' + esc(t) + '</span>';
}).join("");

function fill(el) {
  document.getElementById("q").value = el.textContent;
  document.getElementById("q").focus();
}

function esc(s) {
  return String(s).replace(/[&<>"]/g, function (c) {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
  });
}

function clean(s) {
  // 把连续空白（含换行、制表符）压成单个空格。
  //
  // 为什么需要：PyMuPDF 是按文字位置逐行吐文本的，表格里每个单元格
  // 都会变成独立一行、另外还有大量页眉页脚空行。直接显示就成了
  //     505
  //     -
  //     1,285,752
  //     -
  //     合计
  // 这种「竖排数字粥」，人类根本读不了。
  // 合并成一行后至少是「505 - 1,285,752 - 合计」，能顺着读下去。
  //
  // 注意：只影响**显示**。接口 /search 返回的仍然是原文，
  // 需要逐字核对引用时以接口返回为准。
  return String(s).replace(/\s+/g, " ").trim();
}

async function checkHealth() {
  var pill = document.getElementById("pill");
  try {
    var r = await fetch("/health");
    var h = await r.json();
    document.getElementById("chunks").textContent = h.chunks;
    if (h.status === "ok") {
      pill.className = "pill ok";
      pill.textContent = "服务就绪";
    } else {
      pill.className = "pill loading";
      pill.textContent = "模型加载中…";
      setTimeout(checkHealth, 3000);
    }
  } catch (e) {
    pill.className = "pill loading";
    pill.textContent = "服务未响应";
  }
}

function setBusy(busy) {
  document.getElementById("bask").disabled = busy;
  var g = document.querySelectorAll("button.ghost");
  for (var i = 0; i < g.length; i++) { g[i].disabled = busy; }
}

async function run(mode) {
  var q = document.getElementById("q").value.trim();
  var out = document.getElementById("out");
  if (!q) {
    out.innerHTML = '<div class="card err">请先在上面的输入框里写一个问题。</div>';
    document.getElementById("q").focus();
    return;
  }

  setBusy(true);
  out.innerHTML = '<div class="status"><span class="spinner"></span>' +
    (mode === "ask"
      ? "正在检索并生成答案…（CPU 环境下重排较慢，可能要 10-30 秒）"
      : "正在检索…") + "</div>";

  try {
    var res = await fetch("/" + mode, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, top_k: 5 })
    });
    if (!res.ok) {
      var t = await res.text();
      throw new Error("HTTP " + res.status + " — " + t.slice(0, 200));
    }
    render(mode, await res.json());
  } catch (e) {
    out.innerHTML = '<div class="card err"><b>出错了</b><br>' + esc(e.message) +
      '<br><br>常见原因：服务还在加载模型（看右上角状态），' +
      '或者 .env 里没配 DEEPSEEK_API_KEY。</div>';
  } finally {
    setBusy(false);
  }
}

function render(mode, data) {
  var out = document.getElementById("out");
  var html = "";

  if (mode === "ask") {
    html += '<div class="card' + (data.refused ? " refused" : "") + '">' +
      '<h2>回答 <span class="tag">' +
      (data.refused
        ? "模型判定：知识库里没有依据，已拒答"
        : "方括号数字对应下方证据编号") +
      '</span></h2>' +
      '<div class="answer">' + esc(data.answer) + '</div></div>';
  }

  var hits = data.hits || data.retrieved || [];
  html += '<div class="card"><h2>证据段落 <span class="tag">' +
    '模型实际读到的 ' + hits.length +
    ' 个 chunk（相关度由 cross-encoder 打分；原文换行已合并便于阅读）</span></h2>';

  hits.forEach(function (h) {
    // 不再截断文本：完整内容始终在 DOM 里（可被浏览器的 Ctrl+F 搜到），
    // 只是默认用 CSS 折叠起来，点「展开全文」才显示。
    // 另外把 PDF 带出来的大量换行合并成空格，否则读不了（见 clean 的注释）。
    var flat = clean(h.text);
    var long = flat.length > 180;
    html += '<div class="hit">' +
      '<div class="meta"><span class="rank">' + h.rank + '</span>' +
      '块 #' + h.chunk_id + ' · 相关度 ' + h.score +
      ' · 原文 ' + h.text.length + ' 字</div>' +
      '<div class="txt' + (long ? '' : ' open') + '">' + esc(flat) + '</div>' +
      (long
        ? '<button class="toggle" onclick="toggleText(this)">展开全文 ▾</button>'
        : '') +
      '</div>';
  });

  html += "</div>";
  out.innerHTML = html;
}

function toggleText(btn) {
  var txt = btn.previousElementSibling;
  var opened = txt.classList.toggle("open");
  btn.textContent = opened ? "收起 ▴" : "展开全文 ▾";
}

document.getElementById("q").addEventListener("keydown", function (e) {
  if (e.key === "Enter") { run("ask"); }
});

checkHealth();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def demo():
    """演示页面（给人用的）。

    注意区分：
        /      这个页面，给人看效果的
        /docs  FastAPI 自动生成的 Swagger 调试台，给开发者测接口用的
    """
    return DEMO_HTML