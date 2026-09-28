"""多线程分块下载器 —— 单连接被限速时的解法。

背景（这是个真实的工程坑，值得记一笔）：
    阿里云 pytorch-wheels 镜像单连接只有 ~0.5 MB/s，
    2415 MB 的 CUDA 版 torch 要下 75 分钟。
    但同一镜像开 8 个 Range 并发连接，聚合能到 ~3.7 MB/s，
    说明瓶颈是**单连接限速**，不是总带宽。

    顺带一提：这里用线程而不是进程池 ——
    受限环境下 multiprocessing 需要创建命名管道，会被直接拒绝；
    而下载是 IO 密集型，GIL 在 socket read 时会释放，线程完全够用。

用法：
    python tools/fetch_parallel.py <url> <输出路径> [--threads 8] [--chunk-mb 32]
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

_print_lock = threading.Lock()


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def head_size(url: str) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": UA}, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return int(resp.headers.get("Content-Length") or 0)


def fetch_range(url: str, start: int, end: int, path: Path,
                max_retries: int = 12) -> int:
    """下载 [start, end] 闭区间到 path，支持断点续传。"""
    want = end - start + 1
    attempt = 0
    while attempt < max_retries:
        have = path.stat().st_size if path.exists() else 0
        if have >= want:
            return have
        headers = {"User-Agent": UA, "Range": f"bytes={start + have}-{end}"}
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                mode = "ab" if have else "wb"
                with open(path, mode) as fp:
                    while True:
                        block = resp.read(1 << 20)
                        if not block:
                            break
                        fp.write(block)
                        have += len(block)
            attempt = 0
            if path.stat().st_size >= want:
                return path.stat().st_size
        except Exception as exc:  # noqa: BLE001
            attempt += 1
            with _print_lock:
                print(f"    [!] 分片 {path.name} 第 {attempt} 次中断: "
                      f"{type(exc).__name__}（已有 {human(have)}），续传…", flush=True)
            time.sleep(min(2 * attempt, 10))
    return path.stat().st_size if path.exists() else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="多线程分块下载")
    ap.add_argument("url")
    ap.add_argument("dest")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--chunk-mb", type=int, default=32)
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        print(f"[=] {dest.name} 已存在（{human(dest.stat().st_size)}），跳过")
        return 0

    total = head_size(args.url)
    if not total:
        print("[X] 拿不到 Content-Length，无法分块；改用 tools/fetch_file.py")
        return 1

    chunk = args.chunk_mb * 1024 * 1024
    ranges = [(i, s, min(s + chunk - 1, total - 1))
              for i, s in enumerate(range(0, total, chunk))]

    print(f"目标 {dest.name}")
    print(f"总大小 {human(total)} | {len(ranges)} 个分片 | {args.threads} 线程并发")
    print(f"临时分片目录 {dest.parent}\n")

    done_bytes = 0
    t0 = time.time()
    lock = threading.Lock()

    def task(item):
        idx, start, end = item
        part = dest.with_suffix(dest.suffix + f".p{idx:03d}")
        got = fetch_range(args.url, start, end, part)
        want = end - start + 1
        with lock:
            nonlocal done_bytes
            done_bytes += got
        return idx, got, want, part

    failed: list[int] = []
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = [pool.submit(task, r) for r in ranges]
        for n, fut in enumerate(as_completed(futures), 1):
            idx, got, want, part = fut.result()
            if got < want:
                failed.append(idx)
            elapsed = max(time.time() - t0, 1e-6)
            pct = done_bytes / total * 100
            print(f"  [{n:>2}/{len(ranges)}] 分片{idx:03d} {human(got)}/{human(want)}"
                  f"  总进度 {pct:5.1f}%  {human(done_bytes / elapsed)}/s", flush=True)

    parts = [dest.with_suffix(dest.suffix + f".p{i:03d}") for i, _, _ in ranges]
    if failed or any(not p.exists() or p.stat().st_size != (e - s + 1)
                     for (i, s, e), p in zip(ranges, parts)):
        print(f"\n[X] 有 {len(failed)} 个分片未完成，重跑本命令会自动续传")
        return 1

    print("\n合并分片…")
    with open(dest, "wb") as out:
        for part in parts:
            with open(part, "rb") as fp:
                while True:
                    block = fp.read(1 << 22)
                    if not block:
                        break
                    out.write(block)

    size = dest.stat().st_size
    if size != total:
        print(f"[X] 合并后大小不符：{size} != {total}")
        return 1
    for part in parts:
        part.unlink(missing_ok=True)

    print(f"[OK] {dest}  {human(size)}  用时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
