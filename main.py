import time
import numpy as np
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# -----------------------------
# Mock data: 72 news documents
# -----------------------------
DOCUMENTS = [
    {
        "id": i,
        "content": f"News article {i} discussing economy, climate, and politics.",
        "metadata": {"source": "news"}
    }
    for i in range(72)
]

# -----------------------------
# Embedding utilities (cached)
# -----------------------------
EMBED_DIM = 128

def embed(text: str) -> np.ndarray:
    np.random.seed(abs(hash(text)) % (10**6))
    return np.random.rand(EMBED_DIM)

DOC_EMBEDDINGS = np.vstack([embed(d["content"]) for d in DOCUMENTS])

# Normalize for cosine similarity
DOC_EMBEDDINGS = DOC_EMBEDDINGS / np.linalg.norm(DOC_EMBEDDINGS, axis=1, keepdims=True)

# -----------------------------
# Re-ranking (LLM simulated)
# -----------------------------
def llm_score(query: str, doc: str) -> float:
    # Simulated LLM relevance scoring (0–10)
    score = min(10, max(0, len(set(query.split()) & set(doc.split())) + 3))
    return score / 10.0

# -----------------------------
# Request model
# -----------------------------
class SearchRequest(BaseModel):
    query: str
    k: int = 12
    rerank: bool = True
    rerankK: int = 7

# -----------------------------
# Search endpoint
# -----------------------------
@app.post("/search")
def search(req: SearchRequest):
    start = time.time()

    # Embed query
    q_emb = embed(req.query)
    q_emb = q_emb / np.linalg.norm(q_emb)

    # Vector search
    sims = cosine_similarity([q_emb], DOC_EMBEDDINGS)[0]
    top_idx = sims.argsort()[::-1][:req.k]

    candidates = []
    for i in top_idx:
        candidates.append({
            "id": DOCUMENTS[i]["id"],
            "content": DOCUMENTS[i]["content"],
            "metadata": DOCUMENTS[i]["metadata"],
            "score": float(sims[i])
        })

    # Normalize vector scores to 0–1
    scores = [c["score"] for c in candidates]
    min_s, max_s = min(scores), max(scores)
    for c in candidates:
        c["score"] = (c["score"] - min_s) / (max_s - min_s) if max_s != min_s else 0.5

    reranked = False

    # Re-ranking
    if req.rerank and candidates:
        for c in candidates:
            c["score"] = llm_score(req.query, c["content"])
        candidates.sort(key=lambda x: x["score"], reverse=True)
        candidates = candidates[:req.rerankK]
        reranked = True

    latency = int((time.time() - start) * 1000)

    return {
        "results": candidates,
        "reranked": reranked,
        "metrics": {
            "latency": latency,
            "totalDocs": len(DOCUMENTS)
        }
    }
