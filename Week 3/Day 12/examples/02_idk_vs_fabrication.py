"""
Demo 02: "I Don't Know" vs Fabrication
Block 2 — Run: python code_snippets/02_idk_vs_fabrication.py

Shows: One sentence in the system prompt cuts hallucinations dramatically.
"""
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
)

MODEL = "gpt-4.1-mini"

# Context that does NOT contain contractor information
CONTEXT = """Remote Work Policy (updated January 15, 2024):
Full-time employees may work remotely up to 3 days per week with manager approval.
Employees must be available during core hours (10:00-15:00) on remote days.
Equipment provided: laptop, monitor, keyboard. Internet costs are not reimbursed."""

# Question whose answer is NOT in the context
QUESTION = "What is the company's policy on remote work for contractors?"

# --- System prompt WITHOUT refusal instruction ---
PROMPT_DEFAULT = """You are a helpful assistant answering questions based on provided context.

Context:
{context}"""

# --- System prompt WITH refusal instruction ---
PROMPT_CALIBRATED = """You are a helpful assistant answering questions based on provided context.
Use ONLY the context below to answer. If the answer is not in the context, say "I cannot find this information in the provided documents."

Context:
{context}"""


def run_query(system_prompt, label):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt.format(context=CONTEXT)},
            {"role": "user", "content": QUESTION},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


print(f"Question: {QUESTION}")
print(f"(Answer is NOT in the provided context)")

print(f"\n{'='*60}")
print("=== DEFAULT system prompt (no refusal instruction) ===")
print(f"{'='*60}")
answer_default = run_query(PROMPT_DEFAULT, "default")
print(answer_default)

print(f"\n{'='*60}")
print("=== CALIBRATED system prompt (with refusal instruction) ===")
print(f"{'='*60}")
answer_calibrated = run_query(PROMPT_CALIBRATED, "calibrated")
print(answer_calibrated)

print(f"\n{'='*60}")
print("TAKEAWAY:")
print('Adding "If the answer is not in the context, say I don\'t know"')
print("reduces hallucinations by 30-50% (R-Tuning, 2024).")
print("One sentence. Biggest impact change you can make to a RAG pipeline.")