#!/usr/bin/env python3
"""generate.py

Wraps the generation step: assemble context from retrieved chunks and call the
OpenAI-compatible chat completions endpoint to generate an answer. Returns structured JSON.
"""
import time
from typing import Dict, List

from openai import OpenAI

import config


def generate_answer(query: str, chunks: List[Dict], model: str = None) -> Dict:
    model = model or config.LLM_MODEL
    client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)

    # Build context (simple concatenation)
    context = "\n\n".join([c.get("content", "") for c in chunks])
    prompt = config.SYSTEM_PROMPT.format(context=context) + "\n\nUser question: " + query

    start = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": config.SYSTEM_PROMPT.format(context=context)},
                  {"role": "user", "content": query}],
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
    )
    elapsed = int((time.time() - start) * 1000)

    try:
        content = resp.choices[0].message.content
    except Exception:
        content = str(resp)

    tokens_generated = None
    usage = getattr(resp, "usage", None)
    if usage:
        tokens_generated = usage.get("completion_tokens") if isinstance(usage, dict) else getattr(usage, "completion_tokens", None)

    return {
        "query": query,
        "answer": content,
        "context_used": context,
        "model": model,
        "generation_time_ms": elapsed,
        "tokens_generated": tokens_generated,
    }


if __name__ == "__main__":
    import argparse
    from retrieve import retrieve_chunks

    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    args = parser.parse_args()
    retrieval = retrieve_chunks(args.query)
    print(generate_answer(args.query, retrieval.get("retrieved_chunks", [])))
