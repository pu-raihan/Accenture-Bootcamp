#!/usr/bin/env python3
"""benchmark.py

Run a small benchmark across multiple models and prompts and save results
to a JSON file. Supports interactive subjective rating or non-interactive runs.
"""
import argparse
import json
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

# --- Defaults ---
DEFAULT_MODELS = ["qwen3:4b", "llama3.2:3b"]
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama"
DEFAULT_OUT = "benchmark_results.json"

PROMPTS = [
    {"category": "factual", "prompt": "What causes tides on Earth? Answer in 2-3 sentences."},
    {"category": "reasoning", "prompt": "If a train travels 120km in 1.5 hours, and then 80km in 1 hour, what is the average speed for the entire journey?"},
    {
        "category": "summarization",
        "prompt": "Summarize the following paragraph in one sentence:\n\n" +
        "Artificial intelligence is transforming many industries by enabling automation and improved decision-making. " +
        "Organizations are adopting AI to streamline workflows, reduce costs, and innovate products.",
    },
    {
        "category": "structured",
        "prompt": 'List 3 European capitals. Respond ONLY with valid JSON: [{"city": ..., "country": ...}]',
    },
    {
        "category": "code",
        "prompt": "Write a Python function that returns the fibonacci sequence up to n as a list.",
    },
    {
        "category": "instruction",
        "prompt": "Follow these rules: 1) give a 2-line answer, 2) prefix with 'RESULT:'.\nExplain photosynthesis." ,
    },
]


def run_benchmark(model: str, prompt_text: str, base_url: str, api_key: str, temperature: float = 0.7) -> Dict[str, Any]:
    client = OpenAI(base_url=base_url, api_key=api_key)
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
        temperature=temperature,
    )
    elapsed = time.perf_counter() - start

    # Extract response text safely
    try:
        choice = response.choices[0]
        if hasattr(choice, "message"):
            content = choice.message.get("content") if isinstance(choice.message, dict) else getattr(choice.message, "content", None)
        else:
            content = getattr(choice, "text", None)
    except Exception:
        content = str(response)

    # Usage may be present
    usage = getattr(response, "usage", None)
    prompt_tokens = None
    completion_tokens = None
    tokens_per_second = None
    try:
        if usage:
            prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else getattr(usage, "prompt_tokens", None)
            completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else getattr(usage, "completion_tokens", None)
            if completion_tokens and elapsed > 0:
                tokens_per_second = round(completion_tokens / elapsed, 1)
    except Exception:
        pass

    return {
        "model": model,
        "response": content,
        "time_seconds": round(elapsed, 3),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tokens_per_second": tokens_per_second,
    }


def main(models: List[str], base_url: str, api_key: str, out_file: str, interactive: bool):
    results: List[Dict[str, Any]] = []

    for model in models:
        print(f"Running prompts on model: {model}")
        for p in PROMPTS:
            category = p.get("category")
            prompt_text = p.get("prompt")
            try:
                res = run_benchmark(model=model, prompt_text=prompt_text, base_url=base_url, api_key=api_key)
            except Exception as e:
                res = {"model": model, "response": f"ERROR: {e}", "time_seconds": None}

            record = {
                "model": model,
                "prompt_category": category,
                "prompt": prompt_text,
                "response": res.get("response"),
                "time_seconds": res.get("time_seconds"),
                "prompt_tokens": res.get("prompt_tokens"),
                "completion_tokens": res.get("completion_tokens"),
                "tokens_per_second": res.get("tokens_per_second"),
                "subjective_rating": None,
            }

            print("--- Prompt ---")
            print(prompt_text)
            print("--- Response ---")
            print(record["response"])

            if interactive:
                while True:
                    try:
                        rating = input("Rate quality 1-5 (or Enter to skip): ").strip()
                        if rating == "":
                            break
                        rating_val = int(rating)
                        if 1 <= rating_val <= 5:
                            record["subjective_rating"] = rating_val
                            break
                    except Exception:
                        pass

            results.append(record)

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"models": models, "results": results, "prompts": PROMPTS}, f, indent=2, ensure_ascii=False)

    print(f"Wrote results to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark local LLMs via OpenAI-compatible API (Ollama)")
    parser.add_argument("--models", type=str, default=",")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    parser.add_argument("--interactive", action="store_true", help="Prompt for subjective ratings")

    args = parser.parse_args()

    models_arg = args.models.strip()
    models = DEFAULT_MODELS if models_arg in ("", ",") else [m.strip() for m in models_arg.split(",") if m.strip()]

    main(models=models, base_url=args.base_url, api_key=args.api_key, out_file=args.out, interactive=args.interactive)
