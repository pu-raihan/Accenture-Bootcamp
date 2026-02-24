import json
import os
import time
from deepeval.metrics import (
    AnswerRelevancyMetric, FaithfulnessMetric,
    ContextualRelevancyMetric, ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.models import AzureOpenAIModel

azure_judge = AzureOpenAIModel(
    model="gpt-4.1-mini",
    deployment_name="gpt-4.1-mini",
    api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
    api_version="2025-01-01-preview",
    base_url="https://ds-ai-internship.openai.azure.com",
    timeout=240,
    max_retries=3,
)

with open("pipeline_outputs_local.json") as f:
    results = json.load(f)

test_cases = []
for r in results:
    tc = LLMTestCase(
        input=r["question"],
        actual_output=r["actual_answer"],
        expected_output=r["expected_answer"],
        retrieval_context=r["retrieved_context"],
    )
    test_cases.append(tc)

def make_metrics():
    return [
        FaithfulnessMetric(threshold=0.3, include_reason=True, model=azure_judge),
        AnswerRelevancyMetric(threshold=0.3, include_reason=True, model=azure_judge),
        ContextualRelevancyMetric(threshold=0.3, include_reason=True, model=azure_judge),
        ContextualRecallMetric(threshold=0.3, include_reason=True, model=azure_judge),
        ContextualPrecisionMetric(threshold=0.3, include_reason=True, model=azure_judge),
    ]

BATCH_SIZE = 5
SLEEP_BETWEEN_BATCHES = 30

all_test_results = []  # collect TestResult objects across all batches

for i in range(0, len(test_cases), BATCH_SIZE):
    batch = test_cases[i:i + BATCH_SIZE]
    print(f"Evaluating batch {i // BATCH_SIZE + 1} ({len(batch)} cases)...")
    batch_results = evaluate(batch, make_metrics())  # fresh metrics per batch
    all_test_results.extend(batch_results.test_results)
    if i + BATCH_SIZE < len(test_cases):
        print(f"Sleeping {SLEEP_BETWEEN_BATCHES}s to respect rate limits...")
        time.sleep(SLEEP_BETWEEN_BATCHES)

print("Done. All batches evaluated.")

# Extract detailed results from TestResult objects
output = []

for test_result in all_test_results:
    test_entry = {
        "question": test_result.input,
        "actual_output": test_result.actual_output,
        "expected_output": test_result.expected_output,
        "scores": {},
        "reasons": {},
        "passed": test_result.success,
    }

    for metric_data in test_result.metrics_data:
        name = metric_data.name
        test_entry["scores"][name] = round(metric_data.score, 4) if metric_data.score is not None else None
        test_entry["reasons"][name] = metric_data.reason

    output.append(test_entry)

# Print summary table
print("\n" + "="*80)
print(f"{'Question':<40} {'CtxRel':>7} {'CtxRec':>7} {'CtxPre':>7} {'Faith':>7} {'AnsRel':>7} {'Pass':>6}")
print("-"*80)

for entry in output:
    s = entry["scores"]
    print(
        f"{entry['question'][:38]:<40} "
        f"{str(s.get('Contextual Relevancy', 'N/A')):>7} "
        f"{str(s.get('Contextual Recall', 'N/A')):>7} "
        f"{str(s.get('Contextual Precision', 'N/A')):>7} "
        f"{str(s.get('Faithfulness', 'N/A')):>7} "
        f"{str(s.get('Answer Relevancy', 'N/A')):>7} "
        f"{'✓' if entry['passed'] else '✗':>6}"
    )

# Print failure reasons
print("\n" + "="*80)
print("FAILURE REASONS")
print("="*80)
for entry in output:
    if not entry["passed"]:
        print(f"\nQ: {entry['question']}")
        for metric_name, reason in entry["reasons"].items():
            score = entry["scores"].get(metric_name)
            threshold = 0.3
            status = "✓" if score is not None and score >= threshold else "✗"
            print(f"  {status} {metric_name}: {score} — {reason}")

# Save to JSON
with open("eval_results_cloud.json", "w") as f:
    json.dump(output, f, indent=2)

print("\n✓ Results saved to eval_results_cloud.json")