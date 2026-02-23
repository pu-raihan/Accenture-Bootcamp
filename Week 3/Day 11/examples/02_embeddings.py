"""
Demo 02: Local embeddings with nomic-embed-text
Block 4 — Run: python code_snippets/02_embeddings.py

Shows: Same OpenAI SDK for embeddings. Zero cost per call.
"""
from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.0.114:11436/v1",
    api_key="ollama",
)

texts = [
    "Quantization reduces model size by using fewer bits per weight.",
    "The capital of France is Paris.",
    "Quantized models use INT4 or INT8 instead of FP32.",
]

print("=== Embedding Demo ===\n")

for text in texts:
    response = client.embeddings.create(
        model="nomic-embed-text",
        input=text,
    )
    vector = response.data[0].embedding
    print(f"Text:       \"{text[:60]}...\"")
    print(f"Dimensions: {len(vector)}")
    print(f"First 5:    {[round(v, 4) for v in vector[:5]]}")
    print()

import math

def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)

vecs = []
for text in texts:
    r = client.embeddings.create(model="nomic-embed-text", input=text)
    vecs.append(r.data[0].embedding)

print("=== Similarity Demo ===\n")
print(f"Sim(quantization, Paris):        {cosine_sim(vecs[0], vecs[1]):.4f}")
print(f"Sim(quantization, INT4/INT8):    {cosine_sim(vecs[0], vecs[2]):.4f}")
print(f"\n→ Related texts have higher similarity. This is how RAG retrieval works.")
