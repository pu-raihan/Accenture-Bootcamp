"""
Demo 05: DeepEval — 5 Test Cases, Real Scores
Block 3 — Run: python code_snippets/05_deepeval_demo.py

Shows: DeepEval scores + reasons on cases you can predict.
Requires: uv install deepeval, judge configured (Azure or Ollama)
"""
import os
from deepeval import evaluate
from deepeval.models import AzureOpenAIModel
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric 
from dotenv import load_dotenv

load_dotenv()

JUDGE = AzureOpenAIModel(
        model="gpt-4.1-mini",
        deployment_name="gpt-4.1-mini",
        api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        api_version="2025-01-01-preview",
        base_url="https://ds-ai-internship.openai.azure.com",
    )

test_cases = [
    # 1. Perfect answer — faithful and relevant
    LLMTestCase(
        input="What is the remote work policy?",
        actual_output="Employees may work remotely up to 3 days per week with manager approval.",
        retrieval_context=[
            "Remote Work Policy: Employees may work remotely up to 3 days per week with manager approval. Updated January 2024."
        ],
    ),
    # 2. Hallucinated detail — adds claim not in context
    LLMTestCase(
        input="What is the remote work policy?",
        actual_output="Employees may work remotely up to 5 days per week. The company also provides a $500 monthly stipend for home office equipment.",
        retrieval_context=[
            "Remote Work Policy: Employees may work remotely up to 3 days per week with manager approval. Updated January 2024."
        ],
    ),
    # 3. Wrong date — contradicts context
    LLMTestCase(
        input="When was the policy updated?",
        actual_output="The policy was updated in March 2025.",
        retrieval_context=[
            "Remote Work Policy: Employees may work remotely up to 3 days per week with manager approval. Updated January 2024."
        ],
    ),
    # 4. Off-topic — answers a different question
    LLMTestCase(
        input="How many vacation days do employees get?",
        actual_output="Employees may work remotely up to 3 days per week with manager approval.",
        retrieval_context=[
            "Remote Work Policy: Employees may work remotely up to 3 days per week with manager approval. Updated January 2024."
        ],
    ),
    # 5. Correct refusal — unanswerable question
    LLMTestCase(
        input="What is the policy on contractor remote work?",
        actual_output="I cannot find information about contractor remote work in the provided documents.",
        retrieval_context=[
            "Remote Work Policy: Employees may work remotely up to 3 days per week with manager approval. Updated January 2024."
        ],
    ),
]

LABELS = [
    "Perfect answer",
    "Hallucinated detail",
    "Wrong date",
    "Off-topic answer",
    "Correct refusal",
]

metrics = [
    FaithfulnessMetric(threshold=0.7, model=JUDGE, include_reason=True),
    AnswerRelevancyMetric(threshold=0.7, model=JUDGE, include_reason=True)
]

# --- Run ---
print("Running DeepEval — 5 cases × 2 metrics = 10 judge calls...")
print("(This takes 1-2 minutes with GPT-4.1, longer with local judge)\n")

results = evaluate(test_cases, metrics)

# --- Print results ---
for i, tc in enumerate(test_cases):
    print(f"{'='*60}")
    print(f"Case {i+1}: {LABELS[i]}")
    print(f"  Q: {tc.input}")
    print(f"  A: {tc.actual_output[:80]}{'...' if len(tc.actual_output) > 80 else ''}")

    for metric in metrics:
        metric.measure(tc)
        status = "PASS" if metric.score >= metric.threshold else "FAIL"
        print(f"  {metric.__class__.__name__}: {metric.score:.2f} [{status}]")
        print(f"    → {metric.reason[:120]}")
    print()

print(f"{'='*60}")
print("Notice:")
print("  Case 2: Faithfulness drops — the 'senior directors' claim is fabricated")
print("  Case 3: Faithfulness drops — wrong date contradicts context")
print("  Case 4: Relevancy drops — correct info, wrong question")
print("  Case 5: Both high — refusing is faithful AND relevant")