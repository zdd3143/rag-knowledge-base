"""把分块后的文本编码成向量并存盘。跑一次，以后直接加载。"""
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path("data/chunks.json")
INDEX_DIR = Path("index")


def main():
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    print(f"待编码 {len(chunks)} 块")

    model = SentenceTransformer("BAAI/bge-m3")
    texts = [c["text"] for c in chunks]

    t0 = time.perf_counter()
    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
    )
    print(f"编码耗时 {time.perf_counter() - t0:.1f} 秒")

    INDEX_DIR.mkdir(exist_ok=True)
    np.save(INDEX_DIR / "vectors.npy", vecs.astype(np.float32))
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    (INDEX_DIR / "meta.json").write_text(
        json.dumps({
            "num_chunks": len(chunks),
            "dim": int(vecs.shape[1]),
            "model": "BAAI/bge-m3",
            "normalized": True,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8")

    size_mb = (INDEX_DIR / "vectors.npy").stat().st_size / 1024 / 1024
    print(f"索引已写入 {INDEX_DIR.resolve()}")
    print(f"  {len(chunks)} 块 / {vecs.shape[1]} 维 / {size_mb:.1f} MB")


if __name__ == "__main__":
    main()