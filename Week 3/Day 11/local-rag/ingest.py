#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List

import pypdf
from openai import OpenAI

import config


def extract_text_from_pdf(path: Path) -> List[str]:
    reader = pypdf.PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return pages


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    step = max(1, chunk_size - overlap)
    while i < len(words):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
        i += step
    return chunks


def embed_texts(texts: List[str], client: OpenAI, model: str) -> List[List[float]]:
    embeddings = []
    for t in texts:
        resp = client.embeddings.create(input=t, model=model)
        # compatibility with different client shapes
        if hasattr(resp, "data"):
            vec = resp.data[0].embedding
        else:
            vec = resp["data"][0]["embedding"]
        embeddings.append(vec)
    return embeddings


def persist_to_jsonl(items: List[dict], out_file: Path):
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as fw:
        for it in items:
            fw.write(json.dumps(it, ensure_ascii=False) + "\n")


def try_persist_to_chroma(items: List[dict], collection_name: str) -> bool:
    try:
        import chromadb
        from chromadb.config import Settings

        client = chromadb.Client(Settings())
        # create or get collection
        try:
            collection = client.get_collection(collection_name)
        except Exception:
            collection = client.create_collection(collection_name)

        ids = [it["chunk_id"] for it in items]
        docs = [it["content"] for it in items]
        metadatas = [{"source": it["source"], "page": it["page"]} for it in items]
        embeddings = [it["embedding"] for it in items]

        collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)
        return True
    except Exception:
        return False


def main(docs_dir: str, out_dir: str, use_chroma: bool = True):
    docs_path = Path(docs_dir)
    assert docs_path.exists(), f"Documents path not found: {docs_dir}"

    client = OpenAI(base_url=config.EMBEDDING_BASE_URL, api_key=config.LLM_API_KEY)

    items = []
    for pdf in docs_path.glob("*.pdf"):
        pages = extract_text_from_pdf(pdf)
        for page_no, page_text in enumerate(pages, start=1):
            chunks = chunk_text(page_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            if not chunks:
                continue
            emb_vectors = embed_texts(chunks, client, config.EMBEDDING_MODEL)
            for idx, (chunk, emb) in enumerate(zip(chunks, emb_vectors)):
                items.append(
                    {
                        "source": pdf.name,
                        "page": page_no,
                        "chunk_id": f"{pdf.name}:{page_no}:{idx}",
                        "content": chunk,
                        "embedding": emb,
                    }
                )

    out_path = Path(out_dir)
    # Try ChromaDB first (if requested)
    if use_chroma and try_persist_to_chroma(items, config.COLLECTION_NAME):
        print(f"Persisted {len(items)} items to Chroma collection '{config.COLLECTION_NAME}'")
    else:
        jsonl = out_path / "chunks.jsonl"
        persist_to_jsonl(items, jsonl)
        print(f"Persisted {len(items)} items to {jsonl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs, compute embeddings, store in Chroma or JSONL fallback")
    parser.add_argument("--docs", default="documents", help="Path to documents directory (PDFs)")
    parser.add_argument("--out", default="chroma_db", help="Directory to persist embeddings/chunks")
    parser.add_argument("--no-chroma", action="store_true", help="Do not attempt to write to ChromaDB; use JSONL fallback")
    args = parser.parse_args()
    main(docs_dir=args.docs, out_dir=args.out, use_chroma=not args.no_chroma)
