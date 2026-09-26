from pathlib import Path
import json

def chunk_text(text, size=512, overlap=64):
    """把长文本切成固定长度的小块。
    size:    每块多少字
    overlap: 相邻两块重叠多少字    """
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap      # 每次前进 size-overlap，不是 size
    return chunks


out = []
for txt in Path("data/text").glob("*.txt"):
    text = txt.read_text(encoding="utf-8")
    pieces = chunk_text(text)
    print(f"{txt.name}: {len(text):,} 字 -> {len(pieces)} 块")
    out.append((txt.stem, pieces))

# 存成 JSON 备用
data = [{"doc": name, "chunk_id": f"{name}#{i:05d}", "text": t}
        for name, pieces in out for i, t in enumerate(pieces)]
Path("data/chunks.json").write_text(
    json.dumps(data, ensure_ascii=False), encoding="utf-8")
print(f"总块数：{len(data)}")