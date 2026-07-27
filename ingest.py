"""
Ingest every file in the ./data folder into the vector store.

Run it:   python ingest.py

How it works (the real-world pattern):
  loop over files -> extract text (branch by file type) -> ingest_text()
Your existing ingest_text() already does chunk -> embed -> store, so this
script only adds the "read files and pull text out" layer on top.

Right now it handles .txt and .md (trivial: text is already text).
PDF / DOCX / HTML are stubbed with clear TODOs + the exact code to drop in,
so you can enable them one at a time as you add those file types.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # load .env before importing rag (which reads env vars)

from rag import ingest_text  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def extract_text(path, filename):
    """Return plain text from a file, or None to skip it.
    This is the ONLY format-specific part. Add new file types here."""
    lower = filename.lower()

    # --- Plain text / markdown: already text, just read it ---
    if lower.endswith((".txt", ".md")):
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()

    elif lower.endswith(".pdf"):
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)

    # --- DOCX: enable when you add Word docs ---
    #   pip install python-docx
    # elif lower.endswith(".docx"):
    #     import docx
    #     return "\n".join(p.text for p in docx.Document(path).paragraphs)

    # --- HTML: enable when you add web pages ---
    #   pip install trafilatura   (better than BeautifulSoup for main-content extraction)
    # elif lower.endswith((".html", ".htm")):
    #     import trafilatura
    #     with open(path, encoding="utf-8", errors="ignore") as f:
    #         return trafilatura.extract(f.read())

    # --- PROD SHORTCUT: one library for ALL formats ---
    #   pip install "unstructured[all-docs]"
    #   Replace this whole function's branches with:
    #     from unstructured.partition.auto import partition
    #     return "\n".join(str(el) for el in partition(filename=path))
    #   Trades per-format control for not maintaining a branch per type.

    print(f"  (skipping unsupported file type: {filename})")
    return None


def ingest_folder():
    if not os.path.isdir(DATA_DIR):
        print(f"No data folder at {DATA_DIR}. Create it and add files.")
        return

    files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
    if not files:
        print(f"No files in {DATA_DIR}. Drop your resume.txt (etc.) in there.")
        return

    total = 0
    for filename in files:
        path = os.path.join(DATA_DIR, filename)
        text = extract_text(path, filename)
        if not text or not text.strip():
            continue
        # source=filename -> every chunk remembers which file it came from.
        # This is what lets you cite sources and re-ingest/update one file later.
        n = ingest_text(text, source=filename)
        print(f"  {filename}: {n} chunks")
        total += n

    print(f"\nDone. Ingested {total} chunks from {len(files)} file(s).")


if __name__ == "__main__":
    ingest_folder()