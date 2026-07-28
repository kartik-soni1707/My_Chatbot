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
from google import genai
from google.genai import types
from db import insert_chunks, search_chunks, delete_by_source
from dotenv import load_dotenv
load_dotenv()

# Claude handles GENERATION only (Anthropic has no embedding model).
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
CHAT_MODEL = "gemini-3-flash-preview"# cheap + fast to start; swap for a bigger Claude as needed

# --- Embeddings: Google Gemini (free tier, no local model) ---
gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
EMBEDDING_MODEL = "gemini-embedding-001"
# gemini-embedding-001 defaults to 3072 dims but supports truncation to 768/1536/3072.
# We use 768 (smaller = less storage). MUST match EMBEDDING_DIM in db.py.
EMBEDDING_OUTPUT_DIM = 768

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


# ---------- 2. EMBEDDING (Gemini API) ----------
# Just an HTTP call to Google — no model in our memory. This is the "prod points
# at an embedding model via a key" pattern: light app, provider does the work.
# Note: gemini-embedding-001 accepts ONE text per request, so we loop.
def _embed(text, task_type):
    resp = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBEDDING_OUTPUT_DIM,
        ),
    )
    return resp.embeddings[0].values
 
 
def embed_texts(texts):
    """Embed a list of strings -> list of vectors (for stored documents)."""
    # RETRIEVAL_DOCUMENT tunes the vector for content being stored/searched.
    return [_embed(t, "RETRIEVAL_DOCUMENT") for t in texts]
 
 
def embed_one(text):
    """Embed a single query string -> one vector."""
    # RETRIEVAL_QUERY tunes the vector for a search query (asymmetric to docs).
    return _embed(text, "RETRIEVAL_QUERY")


# ---------- 3. INGEST (offline / preload) ----------
def ingest_text(text, source="manual", replace=True):
    """Chunk a document, embed the chunks, store them. Run this ahead of time.
 
    replace=True (default): delete any existing chunks with the same `source`
    before inserting, so re-ingesting a file REPLACES its chunks instead of
    creating duplicates. This is also how you update a changed document.
    Set replace=False only if you deliberately want to append.
    """
    chunks = chunk_text(text)
    if not chunks:
        return 0
    if replace:
        delete_by_source(source)  # clear this file's old chunks first
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

    # Gemini generation: system prompt goes in config; the user text in contents.
    resp = gemini_client.models.generate_content(
        model=CHAT_MODEL,
        contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=1024,
        ),
    )
    answer = resp.text
    sources = list({src for _c, src, _d in results if src})
    return {"answer": answer, "sources": sources}