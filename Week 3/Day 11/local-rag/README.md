# local-rag — Local Document Q&A (Capstone Day 11)

A complete RAG (Retrieval-Augmented Generation) pipeline using local LLMs via Ollama. Fully offline—no cloud APIs.

## Setup

1. **Create & activate venv:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install openai chromadb langchain langchain-community pypdf
```

2. **Ensure Ollama is running:**
```bash
ollama serve &
```

3. **Pull models (if not already done):**
```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

## Workflow

### 1. Ingest Documents
Extracts text from PDFs, chunks, embeds, and stores in `chroma_db/chunks.jsonl`:
```bash
python "ingest.py" --docs "documents" --out "chroma_db" --no-chroma
```

### 2. Retrieve Chunks
Query the vector database for relevant chunks:
```bash
python "retrieve.py" "What is the refund policy?"
```

Output (JSON):
```json
{
  "query": "...",
  "retrieved_chunks": [
    {"content": "...", "source": "file.pdf", "page": 1, "relevance_score": 0.87}
  ],
  "retrieval_time_ms": 45
}
```

### 3. Generate Answers
End-to-end pipeline: retrieves chunks → sends to LLM → returns structured JSON:
```bash
python "pipeline.py" "What is the refund policy?"
```

Output (JSON):
```json
{
  "query": "...",
  "answer": "According to the documents, ...",
  "context_used": "...",
  "model": "llama3.2:3b",
  "generation_time_ms": 5200,
  "tokens_generated": 42,
  "retrieval": {...}
}
```

### 4. Run Benchmarks
Compare model speeds and quality:
```bash
python "benchmark.py" --models qwen3:4b,llama3.2:3b --interactive
```

## Configuration

Edit `config.py` to change:
- `LLM_MODEL`: which model to use (default: `llama3.2:3b`)
- `EMBEDDING_MODEL`: embedding model (default: `nomic-embed-text`)
- `CHUNK_SIZE`, `CHUNK_OVERLAP`: retrieval granularity
- `TOP_K`: number of chunks to retrieve (default: 5)
- `TEMPERATURE`: LLM response creativity (default: 0.3)

Use environment variables to override (e.g., `LLM_MODEL=qwen3:4b python pipeline.py "..."`).

## Project Structure

```
local-rag/
├── config.py                 # Configuration & environment variables
├── ingest.py                 # PDF → chunks → embeddings → JSONL storage
├── retrieve.py               # Query vector DB, return top-K chunks
├── generate.py               # LLM answer generation
├── pipeline.py               # End-to-end: query → retrieve → generate
├── benchmark.py              # Model comparison benchmark
├── test_questions.json       # 16 test Q&A pairs (5 easy, 5 med, 3 hard, 2 unanswerable)
├── benchmark_results.json    # Benchmark outputs (from Exercise 1)
├── decision_log.md           # Reasoning & tradeoffs
├── README.md                 # This file
├── documents/                # Input PDFs
├── chroma_db/
│   └── chunks.jsonl          # Persisted chunks + metadata (fallback storage)
└── .gitignore
```

## Key Files for Day 12 Evaluation

- **`benchmark_results.json`**: Speed & quality comparison across models
- **`test_questions.json`**: 16 Q&A pairs for evaluation
- **`decision_log.md`**: Architectural decisions & observations
- **Pipeline outputs**: JSON with `retrieval` and `generation` fields for reproducibility

## Notes

- **Speed:** llama3.2:3b is ~14 tok/sec (fast), qwen3:4b is ~4 tok/sec (slower but more detailed)
- **Memory:** Runs on 16GB RAM; may need to `ollama stop <model>` to free memory between runs
- **Retrieval:** Chunks stored in JSONL; if ChromaDB persistence fails, JSONL fallback is automatic
- **Prompt engineering:** System prompt in `config.SYSTEM_PROMPT` forces LLM to use only provided context

## Troubleshooting

- **No chunks retrieved?** Ensure `ingest.py` completed and `chroma_db/chunks.jsonl` exists
- **LLM timeout?** Increase `MAX_TOKENS` or wait longer; qwen3:4b may need >1min per query
- **API not found?** Run `ollama serve` in another terminal

