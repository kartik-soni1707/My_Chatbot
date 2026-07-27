"""
FastAPI wrapper around the RAG pipeline.

Endpoints:
  GET  /health   -> quick check that it's alive
  POST /chat     -> ask a question, get an answer  (this is what your FE calls)

Ingestion is done offline by running `python ingest.py` locally (reads ./data),
so there's no public ingest endpoint — the corpus only changes when you run it.

Start locally:   uvicorn main:app --reload
Start on Render: uvicorn main:app --host 0.0.0.0 --port $PORT
"""
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

from db import init_db          # noqa: E402  (import after load_dotenv)
from rag import answer_question  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs once on startup: make sure the table + extension exist
    init_db()
    yield


app = FastAPI(title="RAG Chatbot", lifespan=lifespan)

# --- CORS: only let your own frontend origins call this from a browser ---
origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is empty")
    return answer_question(req.question)