"""
Demo 03: Context Determines Everything
Block 3 — Run: python code_snippets/03_context_determines.py

Shows: Same question + same model + different context = completely different quality.
RAG quality is a retrieval problem.
"""
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
)

MODEL = "gpt-4.1-mini"
QUESTION = "When was the remote work policy last updated?"

SYSTEM = """You are a helpful assistant. Answer based ONLY on the provided context.

Context:
{context}"""

SCENARIOS = [
    {
        "label": "ACCURATE CONTEXT",
        "context": "Remote Work Policy: Employees may work remotely up to 3 days per week. Policy last updated January 15, 2024.",
        "expected_faithfulness": "1.0 — answer matches context",
    },
    {
        "label": "OUTDATED CONTEXT (wrong year in docs)",
        "context": "Remote Work Policy: Employees may work remotely up to 3 days per week. Policy last updated January 15, 2022.",
        "expected_faithfulness": "Watch — does the model say 2022 (faithful) or 2024 (hallucinated)?",
    },
    {
        "label": "MISSING INFO (no date in context)",
        "context": "Remote Work Policy: Employees may work remotely up to 3 days per week with manager approval. Equipment is provided.",
        "expected_faithfulness": "Should say 'not found' — does it?",
    },
]


for i, scenario in enumerate(SCENARIOS, 1):
    print(f"\n{'='*60}")
    print(f"Scenario {i}: {scenario['label']}")
    print(f"{'='*60}")
    print(f"Context: {scenario['context']}")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM.format(context=scenario["context"])},
            {"role": "user", "content": QUESTION},
        ],
        temperature=0.3,
    )
    answer = response.choices[0].message.content.strip()
    print(f"\nAnswer:  {answer}")
    print(f"Expect:  {scenario['expected_faithfulness']}")

print(f"\n{'='*60}")
print("TAKEAWAY:")
print("The model didn't get smarter or dumber between runs.")
print("The context changed. RAG quality is a retrieval problem.")