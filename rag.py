"""
The actual RAG pipeline. This is the part you'll spend your real energy on.

Flow:
  OFFLINE (once):  chunk -> embed -> store          [ingest_text]
  ONLINE (per query): embed query -> retrieve -> generate   [answer_question]
"""
import os
from openai import OpenAI
from db import insert_chunks, search_chunks

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"  # cheap + good enough to start; swap as you like


# ---------- 1. CHUNKING ----------
# This is the single biggest lever on RAG quality. Start simple, tune later.
# Rough ~500-word chunks with overlap so context isn't lost at boundaries.
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap  # step back by `overlap` for continuity
    return [c for c in chunks if c.strip()]


# ---------- 2. EMBEDDING ----------
def embed_texts(texts):
    """Embed a list of strings -> list of vectors."""
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in resp.data]


def embed_one(text):
    return embed_texts([text])[0]


# ---------- 3. INGEST (offline / preload) ----------
def ingest_text(text, source="manual"):
    """Chunk a document, embed the chunks, store them. Run this ahead of time."""
    chunks = chunk_text(text)
    if not chunks:
        return 0
    embeddings = embed_texts(chunks)
    rows = [(source, chunk, emb) for chunk, emb in zip(chunks, embeddings)]
    insert_chunks(rows)
    return len(rows)


# ---------- 4. RETRIEVE + GENERATE (online / per query) ----------
def answer_question(question, top_k=5):
    # embed the incoming question (cheap, single vector)
    q_emb = embed_one(question)

    # pull the most relevant pre-stored chunks
    results = search_chunks(q_emb, top_k=top_k)
    context = "\n\n---\n\n".join(content for content, _source, _dist in results)

    # build the prompt: retrieved context + the user's question
    system = (
        "You are a helpful assistant. Answer the user's question using ONLY the "
        "context provided. If the answer isn't in the context, say you don't know."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    answer = resp.choices[0].message.content
    sources = list({src for _c, src, _d in results if src})
    return {"answer": answer, "sources": sources}
