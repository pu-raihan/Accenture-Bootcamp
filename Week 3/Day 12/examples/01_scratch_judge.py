"""
Demo 01: Build a Judge from Scratch
Block 2 — Run: python code_snippets/01_scratch_judge.py

Shows: LLM-as-judge is just a prompt.
"""
import json
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
)

JUDGE_PROMPT = """You are evaluating a RAG system answer.

CRITERIA: Faithfulness — does the answer contain ONLY claims supported by the context?
RUBRIC:
  5 = Every claim directly supported by context
  3 = Most claims supported, 1-2 minor unsupported claims
  1 = Answer contains claims that contradict or aren't in context

QUESTION: {question}
CONTEXT: {context}
ANSWER: {answer}

Respond ONLY with valid JSON: {{"score": <1-5>, "reason": "<brief explanation>"}}"""


CONTEXT = "Employees may work remotely up to 3 days per week with manager approval. The policy was updated on January 15, 2024."

TEST_CASES = [
    {
        "label": "FAITHFUL (answer matches context)",
        "question": "What is the remote work policy?",
        "answer": "Employees can work remotely up to 3 days per week, subject to manager approval.",
    },
    {
        "label": "HALLUCINATED (adds facts not in context)",
        "question": "What is the remote work policy?",
        "answer": "Employees can work remotely up to 3 days per week with manager approval. Senior employees in director-level positions may qualify for fully remote arrangements.",
    },
    {
        "label": "CONTRADICTS (wrong date)",
        "question": "When was the remote work policy updated?",
        "answer": "The remote work policy was updated on January 15, 2025.",
    },
]


for i, tc in enumerate(TEST_CASES, 1):
    print(f"\n{'='*60}")
    print(f"Test Case {i}: {tc['label']}")
    print(f"{'='*60}")
    print(f"Answer: {tc['answer']}")

    prompt = JUDGE_PROMPT.format(
        question=tc["question"],
        context=CONTEXT,
        answer=tc["answer"],
    )

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()
    print(f"\nJudge says: {raw}")

    try:
        # Try to parse — model might wrap in markdown
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(cleaned)
        print(f"  Score:  {result['score']}/5")
        print(f"  Reason: {result['reason']}")
    except (json.JSONDecodeError, KeyError):
        print("  (Could not parse JSON — showing raw output above)")

print(f"\n{'='*60}")
print("This is what DeepEval's FaithfulnessMetric does internally —")
print("packaged, with retries, and at scale.")