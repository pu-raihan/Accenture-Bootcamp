# Decision Log — Local RAG Pipeline

## Model Selection
- **Chosen LLM:** llama3.2:3b
- **Why:** 
    - Benchmark results are showing faster token per second for model llama3.2:3b (14.0 tok/s vs 4.3 tok/s).
    - More reliable performance with zero timeouts compared to 1 timeout on qwen3:4b during the code generation test.
    - This is a resource constrained local system (16GB RAM), so this model is suitable for such a deployment as it leaves memory overhead for the vector database and embedding model.
    - For factual tasks, responses are concise and accurate.

- **What I considered but rejected:**
    - **qwen3:4b:** Provides detailed explanations and formatting, but the 3+ minute response times (1.9 tok/s in reasoning) is actually impractical for real-time Q&A. The timeout failure on code generation is a major reliability concern.
    - **qwen3:8b:** Installed initially but rejected because it is too heavy for a 16GB RAM system, causing extreme latency and risk of system crashes during RAG operations.

## Embedding Model
- **Chosen:** nomic-embed-text
- **Why:** 
    - It is lightweight and performs well with the local Ollama API.
    - Designed specifically for semantic search with a large 8192 token context window, which is ideal for processing the 5 PDFs in this project.
    - Compared with all-minilm which is much smaller and faster, but rejected due to the short context window (256–512 tokens). Technical documents have long, complex sentences that would be cut off, leading to poor retrieval quality.

## Chunking Strategy
- **Chunk size:** 512 tokens
- **Overlap:** 50 tokens
- **Why:**
    - Standard balance for document retrieval; 512 tokens (~350 words) is manageable for the LLM context window. 
    -Initially tried Chunk size 256 and overlap 25, but found that technical definitions were being cut off, leading to fragmented answers.
    - 512 is manageable for the LLM context window while keeping more complete information per chunk.
    - 50-token overlap ensures continuity across chunks so that technical context is not lost if a sentence is split between two chunks.

## Retrieval Configuration
- **Top-K:** 5
- **Why:** 
    - Tested Top-K 3, but it often missed secondary supporting facts needed for complex questions.
    - increased to Top-K 5 to provide a better balance between giving the LLM enough information to answer accurately and reducing "noise" from irrelevant text. So it provides enough evidence for the model without hitting memory limits on this local system.

## Observations
- **What worked well:** 
    - Ingestion completed successfully: 85 chunks were persisted to ChromaDB from the provided PDF set.
    - llama3.2:3b outperforms on response speed-14 tok/s- and reliability in this local environment.
    - pipeline architecture successfully separates the ingest then retrieve and then generate steps, allowing for easy model swapping in the `.env` file.
- **What failed:**
    - qwen3:4b timed out on the code generation benchmark, proving it cannot handle high-load tasks on this device.
    - the model sometimes used its own training knowledge instead of relying strictly on the retrieved document context, without strict grounding in the system prompt.
- **Local vs cloud expectations:**
    - Local models are two or 3 times slower than cloud APIs, like Azure OpenAI, but satisfy the industry requirement which is regulated because data never leaves the local infrastructure. Hence provides the ultimate privacy.
    - Trade-off: slower response times in exchange for 100% data privacy and security.

## If I Had More Time / Better Hardware
- Implement a re-ranker to filter the Top-K results for better factual precision further.
- use a higher parameter model (like Llama 3.1 8B) if hardware allowed for 32GB+ RAM.
- Add a streaming response feature to the UI to improve the perceived speed for the user (instinct as a frontend developer)
- Implement hybrid search (combining vector similarity with keyword matching) to better handle specific technical terms.