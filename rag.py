"""
The actual RAG pipeline. This is the part you'll spend your real energy on.

Flow:
  OFFLINE (once):  chunk -> embed -> store          [ingest_text]
  ONLINE (per query): embed query -> retrieve -> generate   [answer_question]

Current setup: LOCAL embeddings (free, no API key) + Claude for generation.
See the PROD notes in the EMBEDDING section for what changes when you move
to a hosted embedding API.
"""
import os
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
from db import insert_chunks, search_chunks
from dotenv import load_dotenv
load_dotenv()

# Claude handles GENERATION only (Anthropic has no embedding model).
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
CHAT_MODEL = "claude-3-5-haiku-latest"  # cheap + fast to start; swap for a bigger Claude as needed


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
# LOCAL model: free, no API key, runs on your CPU. Outputs 384-dim vectors.
# The model downloads once on first run (~90MB), then loads from cache.
_embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ============================ PROD NOTE ============================
# In production you'd typically NOT run the model locally. Instead you'd
# point at a hosted embedding API via a key + model-name config pair, e.g.:
#
#   from openai import OpenAI
#   client = OpenAI(api_key=os.environ["EMBEDDING_API_KEY"])
#   EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]   # e.g. "text-embedding-3-small"
#
#   def embed_texts(texts):
#       resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
#       return [item.embedding for item in resp.data]
#
# Why prod prefers this: no model files/PyTorch/GPU to manage, scales
# effortlessly, provider handles serving. Trade-offs: per-call cost + data
# leaves your machine (which is exactly why privacy/scale cases self-host
# instead, like we're doing locally here).
#
# IMPORTANT when switching: different models output different dimensions
# (MiniLM=384, text-embedding-3-small=1536, voyage=1024). You must update
# EMBEDDING_DIM in db.py to match AND re-embed your whole corpus, because
# vectors from different models live in different spaces and aren't comparable.
# ===================================================================

def embed_texts(texts):
    """Embed a list of strings -> list of 384-float vectors."""
    return _embedder.encode(texts).tolist()


def embed_one(text):
    return _embedder.encode(text).tolist()


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

    # Claude's Messages API: system is its own arg, not a message role.
    resp = client.messages.create(
        model=CHAT_MODEL,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    answer = resp.content[0].text
    sources = list({src for _c, src, _d in results if src})
    return {"answer": answer, "sources": sources}