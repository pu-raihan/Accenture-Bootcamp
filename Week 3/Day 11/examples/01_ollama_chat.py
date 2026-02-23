"""
Demo 01: Chat with a local model via Ollama
Block 4 — Run: python code_snippets/01_ollama_chat.py

Shows: Same OpenAI SDK, local endpoint. Three lines changed from Day 8.
"""
from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.0.114:11436/v1",
    api_key="ollama",  # not checked, but required by SDK
)

response = client.chat.completions.create(
    model="qwen3-vl:2b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "What is quantization in the context of LLMs? Answer in 2 sentences."},
    ],
    temperature=0.7,
    max_tokens=500,
)

print("=== Response Metadata ===")
print(f"Model:          {response.model}")
print(f"Finish reason:  {response.choices[0].finish_reason}")
print(f"Input tokens:   {response.usage.prompt_tokens}")
print(f"Output tokens:  {response.usage.completion_tokens}")
print(f"Total tokens:   {response.usage.total_tokens}")
print()
print("=== The Answer ===")
print(response.choices[0].message.content)
