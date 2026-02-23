# Day 11 Capstone: Build a Local Document Q&A Service

## Scenario

Your client operates in a regulated industry and **cannot send proprietary data to cloud AI providers**. They need an internal document Q&A system that runs entirely on local infrastructure. Your task: build a working RAG pipeline using local LLMs that answers questions about a provided document set, and justify every technical decision you make.

Tomorrow (Day 12), you will formally evaluate this system — measuring retrieval quality, generation accuracy, and security. **Design your pipeline with evaluation in mind.**

---

## What You Will Build

A Python application with three modules:

1. **Ingestion & Retrieval** — Load PDFs, chunk them, embed with a local model, store in a vector database, and retrieve relevant chunks for a query.
2. **Generation** — Take retrieved chunks + user query, generate an answer using a local LLM via OpenAI-compatible API.
3. **Configuration** — A single config file that controls which LLM provider, model, embedding model, and parameters are used. Swapping from local to cloud should require changing only this config.

Plus two artifacts:
- A **benchmark results file** comparing at least 2 models or quantization levels on speed and quality.
- A **decision log** (a simple markdown file) documenting your choices and reasoning.

---

## Phases

| Phase | What to do |
|-------|------------|
| **Setup & Exploration** | Install Ollama, pull models, run benchmarks, explore the API. Complete Exercise 0 and Exercise 1. |
| **Build** | Build the RAG pipeline. Complete Exercise 2. |
| **Document & Submit** | Write your decision log, verify your test questions, push/submit your work. |

---

## Exercise 0: Environment Setup (30 min)

### Install Ollama in WSL

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify it's running:

```bash
ollama --version
ollama serve &    # Start the server in background if not auto-started
```

> **Escape hatch:** If Ollama installation in WSL takes more than 15 minutes or fails, try the native Windows installer from https://ollama.com/download/windows. The API endpoint (`http://localhost:11434`) works the same way.

### Pull the required models

```bash
# Primary models for the capstone
ollama pull qwen3:8b              # Choose one main LLM (~2-5 GB)
ollama pull nomic-embed-text      # Choose one embedding model (~200-500 MB)

# Additional models for comparison (pick at least one)
ollama pull qwen3:4b              # Smaller alternative
ollama pull qwen3-vl:2b           # Smaller alternative
ollama pull phi4-mini             # Microsoft's small reasoning model
ollama pull llama3.2:3b           # Meta's edge model
...
```

> **Memory note:** On 16GB RAM, you can comfortably run one 8B model at a time. If things get slow, close other applications and stick to 3B-4B models. Run `ollama ps` to see what's loaded and `ollama stop <model>` to free memory.

### Set up Python environment

```bash
# Create a project directory
mkdir -p ~/local-rag && cd ~/local-rag

# Option A: Using uv (recommended)
uv init
uv add openai chromadb langchain langchain-community pypdf

# Option B: Using pip + venv (if uv is not available)
python3 -m venv .venv
source .venv/bin/activate
pip install openai chromadb langchain langchain-community pypdf
```

### Verify API access

```python
# test_api.py — Run this to confirm everything works
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

response = client.chat.completions.create(
    model="qwen3:8b",
    messages=[{"role": "user", "content": "Say hello in one sentence."}],
)
print(response.choices[0].message.content)
```

> **If setup took more than 45 minutes** and you still don't have a working Ollama + Python environment: switch to fallback. It's more valuable to spend time on the pipeline architecture than debugging installation issues. You can still score full marks on Exercises 1-2 using the fallback environment.

---

## Exercise 1: Model Comparison & Benchmarking (45–60 min)

**Goal:** Build intuition for the speed-quality-size tradeoff and produce a benchmark artifact.

### Task

Create a script `benchmark.py` that:

1. Defines a list of 5–8 test prompts spanning different capabilities:
   - A factual question
   - A reasoning/math problem
   - A summarization task (provide a paragraph to summarize)
   - A structured output request (e.g., "respond in JSON with fields: ...")
   - A code generation task
   - An instruction-following task with specific format constraints

2. Runs each prompt against at least 2 different models (e.g., `qwen3:4b` and `qwen3:0.6b`, or same model at different quant levels if you downloaded GGUF variants).

3. For each model × prompt combination, records:
   - Response text
   - Tokens per second (from the API response or timing)
   - Total response time
   - Your subjective quality rating (1-5)

4. Outputs results to `benchmark_results.json`.

### Skeleton

```python
# benchmark.py
import json
import time
from openai import OpenAI

# --- Configuration ---
MODELS = ["qwen3:4b", "qwen3-vl:2b"]  # Add/change models here
BASE_URL = "http://localhost:11434/v1"
API_KEY = "ollama"

PROMPTS = [
    {"category": "factual", "prompt": "What causes tides on Earth? Answer in 2-3 sentences."},
    {"category": "reasoning", "prompt": "If a train travels 120km in 1.5 hours, and then 80km in 1 hour, what is the average speed for the entire journey?"},
    {"category": "structured", "prompt": 'List 3 European capitals. Respond ONLY with valid JSON: [{"city": ..., "country": ...}]'},
    # Add more prompts...
]

def run_benchmark(model: str, prompt: str) -> dict:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    elapsed = time.time() - start
    content = response.choices[0].message.content
    usage = response.usage

    return {
        "model": model,
        "response": content,
        "time_seconds": round(elapsed, 2),
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "tokens_per_second": round(usage.completion_tokens / elapsed, 1) if usage and usage.completion_tokens else None,
    }

# --- Main ---
# TODO: Loop over MODELS and PROMPTS, collect results, write to JSON
```

### Deliverable

`benchmark_results.json` — a structured file with all results. You'll reference this in your decision log to justify your model choice for the RAG pipeline.

---

## Exercise 2: Local RAG Pipeline (2.5 hours)

**Goal:** Build a complete document Q&A system that runs entirely locally.

### Documents

Use the document set in `documents/`. This is a set of internal documents from a fictional tech company. Everyone uses the same set so we can compare results after Day 12.

### Architecture Requirements

Your project must have this structure:

```
local-rag/
├── config.py          # All configuration in one place
├── ingest.py          # Document loading, chunking, embedding, storage
├── retrieve.py        # Query the vector DB, return relevant chunks
├── generate.py        # Take chunks + query, produce answer via LLM
├── pipeline.py        # End-to-end: query → retrieve → generate → answer
├── benchmark.py       # From Exercise 1
├── test_questions.json # Your Q&A test set (15-20 pairs)
├── benchmark_results.json
├── decision_log.md    # Your reasoning
├── README.md          # Brief project overview, setup instructions, how to run
├── documents/         # PDFs
└── chroma_db/         # Vector DB storage (gitignored)
```

### config.py — The Provider Swap Layer

This is architecturally important. Tomorrow you'll swap providers to compare local vs cloud.

```python
# config.py
import os

# --- LLM Provider ---
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:8b")

# --- Embedding ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434/v1")

# --- Retrieval ---
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 5
COLLECTION_NAME = "bootcamp_docs"

# --- Generation ---
SYSTEM_PROMPT = """You are a helpful assistant answering questions based on provided context.
Use ONLY the context below to answer. If the answer is not in the context, say "I cannot find this information in the provided documents."

Context:
{context}
"""
MAX_TOKENS = 1024
TEMPERATURE = 0.3
```

### Key Implementation Notes

**For retrieval — log everything.** Your `retrieve.py` should return not just the chunks but metadata about what was retrieved:

```python
# retrieve.py should return something like:
{
    "query": "What is the refund policy?",
    "retrieved_chunks": [
        {
            "content": "Refunds are processed within 30 days...",
            "source": "company_handbook.pdf",
            "page": 12,
            "relevance_score": 0.87
        },
        # ... more chunks
    ],
    "retrieval_time_ms": 45
}
```

This logged output is what Day 12's retrieval evaluation will consume.

**For generation — keep the prompt transparent.** Your `generate.py` should return the full context that was sent to the LLM alongside the answer:

```python
# generate.py should return something like:
{
    "query": "What is the refund policy?",
    "answer": "According to the documents, refunds are processed within 30 days...",
    "context_used": "Refunds are processed within 30 days...",  # What the LLM actually saw
    "model": "qwen3:8b",
    "generation_time_ms": 2340,
    "tokens_generated": 67
}
```

**For the pipeline — wire it together cleanly:**

```python
# pipeline.py
from retrieve import retrieve_chunks
from generate import generate_answer

def answer_question(query: str) -> dict:
    retrieval_result = retrieve_chunks(query)
    generation_result = generate_answer(
        query=query,
        chunks=retrieval_result["retrieved_chunks"]
    )
    return {
        **generation_result,
        "retrieval": retrieval_result,
    }
```

### Test Questions (15-20 pairs)

Create `test_questions.json` with questions and expected answers based on the provided documents:

```json
[
    {
        "id": "q01",
        "question": "What is the maximum number of remote work days per week?",
        "expected_answer": "Employees may work remotely up to 3 days per week with manager approval.",
        "source_document": "company_handbook.pdf",
        "difficulty": "easy",
        "type": "factual"
    },
    {
        "id": "q02",
        "question": "Compare the Standard and Enterprise support tiers.",
        "expected_answer": "Standard offers email support with 48h response time. Enterprise includes 24/7 phone support with 4h response time and a dedicated account manager.",
        "source_document": "product_guide.pdf",
        "difficulty": "medium",
        "type": "comparison"
    },
    {
        "id": "q03",
        "question": "What is the company's position on remote work for contractors?",
        "expected_answer": "The documents do not contain information about contractor remote work policies.",
        "source_document": null,
        "difficulty": "hard",
        "type": "unanswerable"
    }
]
```

**Include at least:**
- 5 easy factual questions (answer is in a single chunk)
- 5 medium questions (require synthesis from multiple chunks or documents)
- 3 hard questions (ambiguous, require reasoning, or have nuanced answers)
- 2 unanswerable questions (answer is NOT in the documents — tests hallucination resistance)

---

## Decision Log Template

Create `decision_log.md` using this structure:

```markdown
# Decision Log — Local RAG Pipeline

## Model Selection
- **Chosen LLM:** [model name and quant level]
- **Why:** [Reference your benchmark results. What did you compare? What made you choose this one?]
- **What I considered but rejected:** [Other models and why they didn't work]

## Embedding Model
- **Chosen:** [model name]
- **Why:** [Did you compare? What was the tradeoff?]

## Chunking Strategy
- **Chunk size:** [value]
- **Overlap:** [value]
- **Why:** [Did you experiment? What happened with different values?]

## Retrieval Configuration
- **Top-K:** [value]
- **Why:** [Did you try different values? What was the impact?]

## Observations
- **What worked well:** [Specific examples from your test questions]
- **What failed:** [Specific examples — which questions did the system get wrong and why?]
- **Local vs cloud expectations:** [How do you think this would compare to Azure OpenAI on the same task?]

## If I Had More Time / Better Hardware
- [What would you change?]
```

---

## Submission

By the end of the session, ensure your project directory contains:
- All Python source files (config, ingest, retrieve, generate, pipeline)
- `benchmark_results.json` from Exercise 1
- `test_questions.json` with 15-20 Q&A pairs
- `decision_log.md` with your reasoning
- `README.md` with setup instructions and how to run the pipeline
- Ingested documents in ChromaDB (or instructions to recreate)

---

## Tips

- **Start with the simplest thing that works.** Get a basic pipeline running with default settings before optimizing. A working system with defaults scores higher than a broken system with "optimal" settings.
- **Memory matters.** On 16GB RAM, running the LLM + embedding model + ChromaDB + your IDE is tight. If things slow down, check memory with `ollama ps`. Stick to 3B-8B models.
- **Test questions are an investment.** The quality of your test set directly affects how useful Day 12's evaluation will be. Don't rush these.
- **Document failures, not just successes.** Your decision log should show what you tried that didn't work. This is more valuable than a list of things that went smoothly.
- **AI coding assistants:** If you use AI coding tools (Copilot, Claude, etc.), that's fine — but you must understand every line in your codebase. If asked "why did you do X?" you need to have an answer beyond "the AI suggested it."
- **Think about Day 12.** Your pipeline outputs need to be machine-readable for evaluation. Structured JSON output is not optional — it's what makes automated evaluation possible.
