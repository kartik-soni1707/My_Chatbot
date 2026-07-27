"""
Database layer: Postgres + pgvector.

This holds your PRE-LOADED document embeddings. The "preload then use" idea:
you ingest + embed your docs once (offline), store the vectors here, and every
query just does a fast similarity search against them.
"""
import os
import psycopg
from pgvector.psycopg import register_vector
from dotenv  import load_dotenv
load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]

# Local model all-MiniLM-L6-v2 returns 384-dimensional vectors.
# MUST match your embedding model's output dimension, or inserts/searches fail.
# PROD NOTE: if you switch to a hosted API model, change this to match it
# (text-embedding-3-small=1536, voyage-3.5=1024) AND re-embed the whole corpus,
# since vectors from different models aren't comparable.
EMBEDDING_DIM = 384


def get_conn():
    """Open a connection with the pgvector type adapter registered."""
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn


def init_db():
    """Create the pgvector extension and the chunks table if they don't exist.
    Run this once at startup."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS doc_chunks (
                    id        BIGSERIAL PRIMARY KEY,
                    source    TEXT,
                    content   TEXT NOT NULL,
                    embedding vector({EMBEDDING_DIM}) NOT NULL
                );
                """
            )
            # Approximate-nearest-neighbour index for fast similarity search.
            # ivfflat needs data to be present to build well; fine for a start.
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS doc_chunks_embedding_idx
                ON doc_chunks
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
                """
            )
        conn.commit()


def insert_chunks(rows):
    """rows: list of (source, content, embedding) tuples."""
    # str(list) -> "[...]" matches pgvector's literal format; ::vector casts it.
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO doc_chunks (source, content, embedding) VALUES (%s, %s, %s::vector);",
                [(source, content, str(emb)) for source, content, emb in rows],
            )
        conn.commit()


def delete_by_source(source):
    """Remove all chunks that came from a given source (filename).
    Used to avoid duplicates: delete a file's old chunks before re-inserting.
    Also how you'd handle a document being updated or removed."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM doc_chunks WHERE source = %s;", (source,))
            deleted = cur.rowcount
        conn.commit()
    return deleted


def search_chunks(query_embedding, top_k=5):
    """Return the top_k most similar chunks to the query embedding.
    Uses cosine distance (<=>). Lower distance = more similar."""
    # str(list) -> "[0.1, 0.2, ...]" which is exactly pgvector's literal format.
    # The ::vector cast tells Postgres to treat it as a vector, not an array.
    vec_literal = str(query_embedding)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT content, source, embedding <=> %s::vector AS distance
                FROM doc_chunks
                ORDER BY distance ASC
                LIMIT %s;
                """,
                (vec_literal, top_k),
            )
            return cur.fetchall()  # list of (content, source, distance)