#!/usr/bin/env python3
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

import config


def _embed_query(query: str, client: OpenAI, model: str) -> List[float]:
    """Embed a query using the embedding model."""
    resp = client.embeddings.create(input=query, model=model)
    if hasattr(resp, "data"):
        return resp.data[0].embedding
    else:
        return resp["data"][0]["embedding"]


def _score_text_match(query: str, text: str) -> float:
    """Simple heuristic: count query words present in text."""
    qwords = set(query.lower().split())
    twords = set(text.lower().split())
    if not qwords:
        return 0.0
    return len(qwords & twords) / len(qwords)


def retrieve_chunks(query: str, top_k: int = None) -> Dict[str, Any]:
    start = time.time()
    top_k = top_k or config.TOP_K
    
    # Try ChromaDB first
    try:
        import chromadb
        from chromadb.config import Settings
        
        client_chroma = chromadb.Client(Settings())
        collection = client_chroma.get_collection(config.COLLECTION_NAME)
        
        # Query by embedding
        client = OpenAI(base_url=config.EMBEDDING_BASE_URL, api_key=config.LLM_API_KEY)
        query_emb = _embed_query(query, client, config.EMBEDDING_MODEL)
        
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        retrieved = []
        if results and results["documents"]:
            for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
                # Chroma distances are euclidean by default; lower = more similar
                # Convert to similarity score (0-1, higher = more similar)
                score = 1.0 / (1.0 + dist) if dist >= 0 else 1.0
                retrieved.append({
                    "content": doc,
                    "source": meta.get("source"),
                    "page": meta.get("page"),
                    "relevance_score": float(score),
                })
        
        return {"query": query, "retrieved_chunks": retrieved, "retrieval_time_ms": int((time.time() - start) * 1000)}
    except Exception:
        pass
    
    # Fallback to JSONL word matching
    dbfile = Path("chroma_db/chunks.jsonl")
    retrieved = []

    if not dbfile.exists():
        return {"query": query, "retrieved_chunks": [], "retrieval_time_ms": int((time.time() - start) * 1000)}

    candidates = []
    with dbfile.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            score = _score_text_match(query, item.get("content", ""))
            candidates.append((score, item))

    candidates.sort(key=lambda x: x[0], reverse=True)
    for score, item in candidates[:top_k]:
        retrieved.append({
            "content": item.get("content"),
            "source": item.get("source"),
            "page": item.get("page"),
            "relevance_score": float(score),
        })

    return {"query": query, "retrieved_chunks": retrieved, "retrieval_time_ms": int((time.time() - start) * 1000)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    args = parser.parse_args()
    result = retrieve_chunks(args.query)
    import json as json_mod
    print(json_mod.dumps(result, indent=2, ensure_ascii=False))
