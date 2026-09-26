from pathlib import Path
import fitz

raw_dir = Path("data/raw")
out_dir = Path("data/text")
out_dir.mkdir(parents=True, exist_ok=True)

for pdf in raw_dir.glob("*.pdf"):
    print(f"处理 {pdf.name} ...", end=" ")
    doc = fitz.open(pdf)
    pages = [page.get_text("text") for page in doc]
    doc.close()

    text = "\n".join(pages)
    (out_dir / (pdf.stem + ".txt")).write_text(text, encoding="utf-8")
    print(f"{len(pages)} 页 / {len(text):,} 字")