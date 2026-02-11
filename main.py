import math
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import httpx

# -----------------------------
# Configuration
# -----------------------------
# OPENAI_BASE_URL = "https://api.openai.com/v1"
# OPENAI_API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIyZjIwMDA5ODRAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.G7srIOp35q_kYBkoQ9D4CusHekbXlHbCvsP4YiuaoRM"  # replace with your key
# EMBEDDING_MODEL = "text-embedding-3-small"

OPENAI_BASE_URL = "https://aipipe.org/openai/v1"  # if using aipipe
OPENAI_API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIyZjIwMDA5ODRAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.G7srIOp35q_kYBkoQ9D4CusHekbXlHbCvsP4YiuaoRM"
EMBEDDING_MODEL = "text-embedding-3-small"


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["OPTIONS", "POST"],
    allow_headers=["*"],
)

# -----------------------------
# Request model
# -----------------------------
class SimilarityRequest(BaseModel):
    docs: List[str]
    query: str

# -----------------------------
# Utility functions
# -----------------------------
def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)

# async def get_embeddings(texts: List[str]) -> List[List[float]]:
async def get_embeddings(texts):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{OPENAI_BASE_URL}/embeddings",
            json=payload,
            headers=headers,
            timeout=30
        )
        response.raise_for_status()

    return [item["embedding"] for item in response.json()["data"]]

# -----------------------------
# API endpoint
# -----------------------------
@app.post("/similarity")
async def similarity(req: SimilarityRequest):
    embeddings = await get_embeddings(req.docs + [req.query])

    doc_embeddings = embeddings[:-1]
    query_embedding = embeddings[-1]

    scored = []
    for doc, emb in zip(req.docs, doc_embeddings):
        score = cosine_similarity(query_embedding, emb)
        scored.append((doc, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    return {
        "matches": [doc for doc, _ in scored[:3]]
    }

