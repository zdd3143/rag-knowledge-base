"""阶段 9：把 demo 变成能上生产的三个必备件。

一个能跑的 RAG demo 和一个能上线的 RAG 服务，差的主要就是这三件事：

    ① 成本可观测   —— 不知道每次问答花多少钱，就没法做预算、也没法做优化
    ② 查询缓存     —— 相同问题重复问，没必要重复烧 token
    ③ 重试与降级   —— 大模型 API 会超时、会 429、会 5xx，
                      不能让它把整个请求打挂

这三个是「AI 应用工程师」和「跑通 demo 的人」的分水岭 ——
招聘 JD 里写的「成本管控 / 权限 / 日志 / 异常处理 / 监控告警」，
核心就是这一层。

设计原则：**做成可插拔的中间件，不改动原有检索逻辑。**
    stage6_generate.ask_llm()   原来的裸调用（保持不动，方便对照）
    stage9_production.MeteredLLM  包一层，加上三件事
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import requests

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

# ============================================================
# ⚠️ 价格是【可配置】的，不要硬编码在逻辑里
# ------------------------------------------------------------
# 大模型定价变动很频繁（而且不同渠道价格不同：
# 官方 API / 云厂商代理 / 包年 —— 能差好几倍）。
# 所以：
#   · 价格放在这里，一处修改全局生效
#   · 支持用环境变量覆盖（部署时不用改代码）
#   · 数值以【官方定价页】为准，下面只是 2026-10 的参考值
# ============================================================
DEFAULT_PRICE = {
    # 单位：元 / 百万 token
    "deepseek-chat": {"input": 2.0, "output": 8.0},
}


def _price_for(model: str) -> dict:
    override = os.environ.get("LLM_PRICE_JSON")
    if override:
        try:
            return json.loads(override).get(model, DEFAULT_PRICE["deepseek-chat"])
        except json.JSONDecodeError:
            pass
    return DEFAULT_PRICE.get(model, DEFAULT_PRICE["deepseek-chat"])


# ============================================================
# ① 成本可观测
# ============================================================
@dataclass
class Usage:
    """累计用量。可以随时 asdict() 出去挂到 /metrics 上。"""
    calls: int = 0              # 真正打到 API 的次数（不含缓存命中）
    cache_hits: int = 0         # 缓存命中次数（省下的调用）
    retries: int = 0            # 重试次数
    failures: int = 0           # 最终失败的次数
    degraded: int = 0           # 走了降级分支的次数
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_yuan: float = 0.0
    latency_ms_total: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def avg_latency_ms(self) -> float:
        return self.latency_ms_total / self.calls if self.calls else 0.0

    @property
    def cache_hit_rate(self) -> float:
        asked = self.calls + self.cache_hits
        return self.cache_hits / asked if asked else 0.0

    @property
    def saved_yuan(self) -> float:
        """缓存【估算】省下的钱。

        ⚠️ 这是估算：用「已发生的平均单次成本 × 命中次数」推的。
        真实省下的钱取决于那些被缓存的问题本来会花多少 —— 无法反事实测量。
        所以对外汇报时要说清这是 estimate，不能当成精确账目。
        """
        if not self.cache_hits or not self.calls:
            return 0.0
        return self.cost_yuan / self.calls * self.cache_hits


class TokenMeter:
    """记录 token 用量与成本。线程安全（FastAPI 是多线程的）。"""

    def __init__(self, model: str = "deepseek-chat"):
        self.model = model
        self.usage = Usage()
        self._lock = threading.Lock()

    def record(self, prompt_tokens: int, completion_tokens: int,
               latency_ms: int):
        price = _price_for(self.model)
        cost = (prompt_tokens / 1e6 * price["input"]
                + completion_tokens / 1e6 * price["output"])
        with self._lock:
            self.usage.calls += 1
            self.usage.prompt_tokens += prompt_tokens
            self.usage.completion_tokens += completion_tokens
            self.usage.cost_yuan += cost
            self.usage.latency_ms_total += latency_ms
        return cost

    def bump(self, **kw):
        """给 cache_hits / retries / failures / degraded 加计数。"""
        with self._lock:
            for k, v in kw.items():
                setattr(self.usage, k, getattr(self.usage, k) + v)

    def snapshot(self) -> dict:
        with self._lock:
            d = asdict(self.usage)
        d["total_tokens"] = self.usage.total_tokens
        d["avg_latency_ms"] = round(self.usage.avg_latency_ms, 1)
        d["cache_hit_rate"] = round(self.usage.cache_hit_rate, 4)
        d["saved_yuan_estimate"] = round(self.usage.saved_yuan, 6)
        d["cost_yuan"] = round(d["cost_yuan"], 6)
        return d


# ============================================================
# ② 查询缓存
# ============================================================
class QueryCache:
    """LRU + TTL 的问答缓存。

    ⭐⭐ 关键设计：key 必须包含【检索配置的指纹】

    这是最容易踩的坑：如果只拿问题文本当 key，
    那么你把 top_k 从 5 改成 8、或者换了 embedding 模型、改了融合权重，
    **缓存里的旧答案还会被返回** —— 而且你完全不会察觉，
    因为结果看起来完全正常（就是"旧配置的正确答案"）。

    ⇒ 所以 key = hash(问题 + 检索配置指纹)。
      配置一变，指纹变，缓存自动失效。**这是把「静默错误」变成「不会发生」。**

    另外两个细节：
      · LRU：内存有限，按最近使用淘汰（不用 FIFO，因为热点问题会反复问）
      · TTL：大模型会更新、文档会更新，缓存不能永不过期
    """

    def __init__(self, maxsize: int = 500, ttl_seconds: int = 3600,
                 config_fingerprint: str = ""):
        self.maxsize = maxsize
        self.ttl = ttl_seconds
        self.fingerprint = config_fingerprint
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def make_key(self, question: str) -> str:
        """问题 + 配置指纹 → 缓存键。

        用 sha256 而不是 md5：不是为了安全，是为了**跨环境一致**
        （有些环境的 md5 实现受 FIPS 限制会报错，sha256 到处都有）。
        """
        raw = f"{self.fingerprint}\x00{question.strip()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def get(self, question: str):
        key = self.make_key(question)
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self.misses += 1
                return None
            ts, value = item
            if time.time() - ts > self.ttl:
                # 过期：删掉并算 miss
                self._data.pop(key, None)
                self.misses += 1
                return None
            self._data.move_to_end(key)      # LRU：标记为最近使用
            self.hits += 1
            return value

    def put(self, question: str, value: Any):
        key = self.make_key(question)
        with self._lock:
            self._data[key] = (time.time(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)   # 淘汰最久未用

    def clear(self):
        with self._lock:
            self._data.clear()
            self.hits = self.misses = 0

    @property
    def size(self) -> int:
        return len(self._data)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


# ============================================================
# ③ 重试与降级
# ============================================================
class RetryPolicy:
    """重试策略。

    ⭐ 最重要的一条：**区分「可重试」和「不可重试」**

        可重试（重试有意义）：
            429 限流、500/502/503/504 服务端错误、连接超时/读超时
        不可重试（重试只是浪费时间、还可能重复扣费）：
            400 请求体错误、401 鉴权失败、403 无权限 —— 这些重试一百次也一样

    ⭐ 第二条：**指数退避 + 随机抖动**

        退避：等 1s → 2s → 4s。因为如果对方在限流，你立刻重试只会加剧。
        抖动：加 ±30% 随机。否则所有客户端会在同一时刻一起重试
              （thundering herd），把刚恢复的服务再打挂。

    ⭐ 第三条：**重试要有上限，最终必须降级**

        重试到底还是失败的话，不能把异常抛给用户 ——
        要返回一句明确的话（"服务繁忙，请稍后再试"），
        并且**记录下来**。静默失败和静默成功一样危险。
    """

    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0,
                 max_delay: float = 8.0, jitter: float = 0.3):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter

    @staticmethod
    def is_retryable(exc: Exception) -> bool:
        if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
            return True
        if isinstance(exc, requests.HTTPError) and exc.response is not None:
            code = exc.response.status_code
            return code == 429 or 500 <= code < 600
        return False

    def sleep_for(self, attempt: int) -> float:
        """第 attempt 次失败后该等多久（attempt 从 1 开始）。"""
        delay = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
        # 抖动：在 [1-j, 1+j] 区间随机
        return delay * (1 + random.uniform(-self.jitter, self.jitter))


def call_with_retry(fn: Callable[[], Any], policy: RetryPolicy | None = None,
                    on_retry: Callable[[int, Exception, float], None] | None = None,
                    fallback: Any = None) -> tuple[Any, dict]:
    """带重试和降级的调用。返回 (结果, 元信息)。

    元信息里记录 attempts / retried / degraded —— 这些能挂到监控上，
    用来回答"我们的模型服务健康吗"。
    """
    policy = policy or RetryPolicy()
    last_exc: Exception | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn(), {"attempts": attempt, "retried": attempt > 1,
                          "degraded": False, "error": None}
        except Exception as exc:                      # noqa: BLE001
            last_exc = exc
            if not policy.is_retryable(exc) or attempt == policy.max_attempts:
                break
            delay = policy.sleep_for(attempt)
            if on_retry:
                on_retry(attempt, exc, delay)
            time.sleep(delay)

    # 走到这里说明：不可重试，或者重试次数用完了
    info = {"attempts": policy.max_attempts
            if last_exc and policy.is_retryable(last_exc) else attempt,
            "retried": True, "degraded": True,
            "error": f"{type(last_exc).__name__}: {last_exc}"}
    return fallback, info


# ============================================================
# 组合：一个带成本、缓存、重试的 LLM 客户端
# ============================================================
class MeteredLLM:
    """把三件事组合起来。对外只暴露一个 .ask()。

    调用顺序（这个顺序很重要）：
        缓存 → 重试 → 计量

    · 先查缓存：命中就完全不花钱
    · 未命中才走重试逻辑
    · 成功后才计量（失败的调用通常不计费，但重试次数要记）
    """

    def __init__(self, api_key: str, model: str = "deepseek-chat",
                 cache: QueryCache | None = None,
                 policy: RetryPolicy | None = None,
                 session: requests.Session | None = None):
        self.api_key = api_key
        self.model = model
        self.meter = TokenMeter(model)
        self.cache = cache or QueryCache()
        self.policy = policy or RetryPolicy()
        self.session = session or requests.Session()
        self.call_log: list[dict] = []          # 最近的调用记录（给 /metrics 用）

    # ---------- 内部：真正发请求 ----------
    def _post(self, system_prompt: str, user_prompt: str,
              max_tokens: int, timeout: int):
        resp = self.session.post(
            DEEPSEEK_URL,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [{"role": "system", "content": system_prompt},
                             {"role": "user", "content": user_prompt}],
                "temperature": 0.1,
                "max_tokens": max_tokens,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()

    # ---------- 对外 ----------
    def ask(self, question: str, system_prompt: str, user_prompt: str,
            max_tokens: int = 1024, timeout: int = 120) -> dict:
        # ---- ① 缓存 ----
        cached = self.cache.get(question)
        if cached is not None:
            self.meter.bump(cache_hits=1)
            return {**cached, "from_cache": True, "cost_yuan": 0.0}

        # ---- ② 重试 ----
        t0 = time.perf_counter()
        data, info = call_with_retry(
            lambda: self._post(system_prompt, user_prompt, max_tokens, timeout),
            self.policy,
            on_retry=lambda a, e, d: self.meter.bump(retries=1),
        )
        elapsed = int((time.perf_counter() - t0) * 1000)

        # ---- ③ 降级 ----
        if info["degraded"]:
            self.meter.bump(failures=1, degraded=1)
            return {
                "answer": "抱歉，模型服务暂时不可用，请稍后再试。",
                "degraded": True, "from_cache": False,
                "error": info["error"], "attempts": info["attempts"],
                "cost_yuan": 0.0,
            }

        # ---- ④ 计量 ----
        usage = data.get("usage") or {}
        pt = int(usage.get("prompt_tokens", 0))
        ct = int(usage.get("completion_tokens", 0))
        cost = self.meter.record(pt, ct, elapsed)

        answer = data["choices"][0]["message"]["content"].strip()
        result = {
            "answer": answer,
            "degraded": False, "from_cache": False,
            "prompt_tokens": pt, "completion_tokens": ct,
            "latency_ms": elapsed, "attempts": info["attempts"],
            "cost_yuan": round(cost, 6),
        }
        self.cache.put(question, {k: v for k, v in result.items()
                                  if k != "from_cache"})
        self.call_log.append({"q": question[:40], "ms": elapsed,
                              "tokens": pt + ct, "cost": round(cost, 6),
                              "at": time.strftime("%H:%M:%S")})
        self.call_log[:] = self.call_log[-50:]
        return result

    def metrics(self) -> dict:
        return {
            "usage": self.meter.snapshot(),
            "cache": {"size": self.cache.size,
                      "hit_rate": round(self.cache.hit_rate, 4),
                      "maxsize": self.cache.maxsize,
                      "ttl_seconds": self.cache.ttl,
                      "fingerprint": self.cache.fingerprint[:12]},
            "recent_calls": self.call_log[-10:],
        }


# ============================================================
# 自检 —— 不用真的调 API 就能验证三件事都工作
# ============================================================


def _fake_response(prompt_tokens=800, completion_tokens=120):
    class R:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"choices": [{"message": {"content": "这是答案 [1]"}}],
                    "usage": {"prompt_tokens": prompt_tokens,
                              "completion_tokens": completion_tokens}}
    return R()


class _FakeSession:
    """模拟一个"前 N 次失败、之后成功"的 API。"""

    def __init__(self, fail_times=0, fail_with=503):
        self.n = 0
        self.fail_times = fail_times
        self.fail_with = fail_with

    def post(self, *a, **kw):
        self.n += 1
        if self.n <= self.fail_times:
            err = requests.HTTPError(f"{self.fail_with} 服务端错误")
            resp = requests.Response()
            resp.status_code = self.fail_with
            err.response = resp
            raise err
        return _fake_response()


def main():
    print("=" * 74)
    print("阶段 9：生产必备的三件事 —— 成本可观测 / 缓存 / 重试降级")
    print("=" * 74)

    # ---------- ① 成本可观测 ----------
    print("\n【① 成本可观测】")
    m = TokenMeter("deepseek-chat")
    # 模拟 100 次问答：每次输入 800 token、输出 120 token
    for _ in range(100):
        m.record(800, 120, latency_ms=1600)
    s = m.snapshot()
    print(f"  模拟 100 次问答（每次 in 800 / out 120 token）")
    print(f"     总 token     {s['total_tokens']:,}")
    print(f"     总成本       ¥{s['cost_yuan']:.4f}")
    print(f"     单次成本     ¥{s['cost_yuan'] / 100:.6f}")
    print(f"     平均延迟     {s['avg_latency_ms']} ms")
    print(f"  ⭐ 意义：知道单次成本，才能算「1000 个员工每天问 10 次」要多少钱")

    # ---------- ② 查询缓存 ----------
    print("\n【② 查询缓存】")
    c = QueryCache(maxsize=3, ttl_seconds=60, config_fingerprint="topk=5|bge-m3|0.3/0.7")
    c.put("中国石油2023年的营业收入是多少？", {"answer": "3.01 万亿元"})
    hit = c.get("中国石油2023年的营业收入是多少？")
    print(f"  第一次问 → {'命中' if hit else '未命中'}  {hit}")
    print(f"  命中率 {c.hit_rate:.0%}   缓存条数 {c.size}")

    # ⭐ 核心演示：改检索配置 → 指纹变 → 缓存自动失效
    c2 = QueryCache(config_fingerprint="topk=5|bge-m3|0.3/0.7")
    c2.put("同一个问题", "旧配置的答案")
    c2.fingerprint = "topk=8|bge-m3|0.3/0.7"     # 改了 top_k
    print(f"\n  ⭐ 改了检索配置后，同一个问题：")
    print(f"     命中缓存吗？ {c2.get('同一个问题')}   ← 自动失效，不会返回旧答案")
    print(f"     （如果 key 里不带配置指纹，这里会静默返回旧配置的答案）")

    # LRU 淘汰
    c3 = QueryCache(maxsize=2)
    for q in ["A", "B", "C"]:
        c3.put(q, f"答案{q}")
    print(f"\n  LRU 淘汰：maxsize=2 时插入 A/B/C → 还剩 "
          f"{[c3.get('A'), c3.get('B'), c3.get('C')]}")
    print(f"     （A 被淘汰了 —— 因为它最久没用）")

    # ---------- ③ 重试与降级 ----------
    print("\n【③ 重试与降级】")
    pol = RetryPolicy(max_attempts=3, base_delay=0.01, max_delay=0.05)

    # 场景 1：前 2 次 503，第 3 次成功 → 应该重试后成功
    fs = _FakeSession(fail_times=2, fail_with=503)
    llm = MeteredLLM("fake-key", session=fs, policy=pol,
                     cache=QueryCache())
    r = llm.ask("q1", "sys", "user")
    print(f"  场景1  前 2 次 503 后成功：")
    print(f"         attempts={r['attempts']}  degraded={r['degraded']}  "
          f"重试计数={llm.meter.usage.retries}  ✅")

    # 场景 2：一直 503 → 重试用完 → 降级
    fs2 = _FakeSession(fail_times=99, fail_with=503)
    llm2 = MeteredLLM("fake-key", session=fs2, policy=pol, cache=QueryCache())
    r2 = llm2.ask("q2", "sys", "user")
    print(f"  场景2  一直 503：")
    print(f"         degraded={r2['degraded']}  回复={r2['answer'][:22]}…  "
          f"失败计数={llm2.meter.usage.failures}  ✅")

    # 场景 3：401 不可重试 → 立刻降级，不浪费时间
    fs3 = _FakeSession(fail_times=99, fail_with=401)
    llm3 = MeteredLLM("bad-key", session=fs3, policy=pol, cache=QueryCache())
    r3 = llm3.ask("q3", "sys", "user")
    print(f"  场景3  401 鉴权失败（不可重试）：")
    print(f"         实际请求次数={fs3.n}（应该是 1，没有浪费重试）  "
          f"degraded={r3['degraded']}  ✅")

    # 场景 4：缓存命中 → 完全不花钱
    fs4 = _FakeSession()
    llm4 = MeteredLLM("k", session=fs4, policy=pol, cache=QueryCache())
    a = llm4.ask("同样的问题", "sys", "user")
    b = llm4.ask("同样的问题", "sys", "user")
    print(f"  场景4  同一个问题问 2 次：")
    print(f"         第1次 from_cache={a['from_cache']}  花费 ¥{a['cost_yuan']}")
    print(f"         第2次 from_cache={b['from_cache']}  花费 ¥{b['cost_yuan']} "
          f"（省下来了）")
    print(f"         实际打到 API 次数 = {fs4.n}（应该是 1）  ✅")

    print("\n" + "=" * 74)
    print("结论：三件事都做成可插拔中间件，检索逻辑一行没动。")
    print("=" * 74)


if __name__ == "__main__":
    main()
