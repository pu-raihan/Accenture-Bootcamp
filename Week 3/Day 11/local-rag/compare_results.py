#!/usr/bin/env python3
"""
Compare local vs cloud evaluation results.
Builds a summary table and analysis.
"""
import json
import statistics

def load_results(filename):
    """Load evaluation results from JSON."""
    with open(filename) as f:
        return json.load(f)

def load_pipeline_outputs(filename):
    """Load pipeline outputs to extract timing/metadata."""
    with open(filename) as f:
        return json.load(f)

# Load all results
print("Loading results...\n")
eval_local = load_results("eval_results_local.json")
eval_cloud = load_results("eval_results_cloud.json")
pipeline_local = load_pipeline_outputs("pipeline_outputs_local.json")
pipeline_cloud = load_pipeline_outputs("pipeline_outputs_cloud.json")

# Extract metric scores
def get_metric_stats(eval_results):
    """Extract min/max/avg for each metric."""
    metrics = {
        "Faithfulness": [],
        "Answer Relevancy": [],
        "Contextual Relevancy": [],
        "Contextual Recall": [],
        "Contextual Precision": [],
    }
    
    for entry in eval_results:
        for metric_name, metric_data in entry["scores"].items():
            if metric_data is not None:
                metrics[metric_name].append(metric_data)
    
    stats = {}
    for metric_name, scores in metrics.items():
        if scores:
            stats[metric_name] = {
                "avg": round(statistics.mean(scores), 4),
                "min": round(min(scores), 4),
                "max": round(max(scores), 4),
                "count": len(scores),
            }
    return stats

# Get timing stats
def get_timing_stats(pipeline_outputs):
    """Extract average retrieval and generation times."""
    retrieval_times = [r.get("retrieval_time_ms", 0) for r in pipeline_outputs if r.get("model") != "error"]
    generation_times = [r.get("generation_time_ms", 0) for r in pipeline_outputs if r.get("model") != "error"]
    
    return {
        "avg_retrieval_ms": round(statistics.mean(retrieval_times), 0) if retrieval_times else 0,
        "avg_generation_ms": round(statistics.mean(generation_times), 0) if generation_times else 0,
    }

# Get pass rates
def get_pass_rates(eval_results):
    """Get pass rates per metric."""
    pass_rates = {}
    for metric_name in ["Faithfulness", "Answer Relevancy", "Contextual Relevancy", "Contextual Recall", "Contextual Precision"]:
        passes = sum(1 for e in eval_results if e["scores"].get(metric_name, 0) and e["scores"][metric_name] >= 0.3)
        total = sum(1 for e in eval_results if e["scores"].get(metric_name) is not None)
        pass_rates[metric_name] = round(passes / total * 100, 1) if total > 0 else 0
    return pass_rates

stats_local = get_metric_stats(eval_local)
stats_cloud = get_metric_stats(eval_cloud)
timing_local = get_timing_stats(pipeline_local)
timing_cloud = get_timing_stats(pipeline_cloud)
pass_rates_local = get_pass_rates(eval_local)
pass_rates_cloud = get_pass_rates(eval_cloud)

print("="*100)
print("LOCAL vs CLOUD EVALUATION COMPARISON")
print("="*100)
print()

# Main comparison table
print(f"{'Metric':<30} {'Local Avg':>12} {'Cloud Avg':>12} {'Delta':>12} {'Local Pass%':>12} {'Cloud Pass%':>12}")
print("-"*100)

for metric in ["Faithfulness", "Answer Relevancy", "Contextual Relevancy", "Contextual Recall", "Contextual Precision"]:
    local_avg = stats_local[metric]["avg"]
    cloud_avg = stats_cloud[metric]["avg"]
    delta = round(cloud_avg - local_avg, 4)
    
    local_pass = pass_rates_local[metric]
    cloud_pass = pass_rates_cloud[metric]
    
    delta_str = f"+{delta}" if delta >= 0 else f"{delta}"
    print(f"{metric:<30} {local_avg:>12.4f} {cloud_avg:>12.4f} {delta_str:>12} {local_pass:>11.1f}% {cloud_pass:>11.1f}%")

print()
print(f"Avg Response Time (retrieval + generation):")
print(f"  Local: {timing_local['avg_retrieval_ms']:.0f}ms (retrieval) + {timing_local['avg_generation_ms']:.0f}ms (generation)")
print(f"  Cloud: {timing_cloud['avg_retrieval_ms']:.0f}ms (retrieval) + {timing_cloud['avg_generation_ms']:.0f}ms (generation)")
print()

# Find hallucinations (unanswerable questions where model didn't say "I cannot find")
print("="*100)
print("HALLUCINATION ANALYSIS (Questions where answer should have been 'I cannot find...')")
print("="*100)

def find_hallucinations(eval_results, pipeline_outputs):
    """Find cases where model answered unanswerable questions."""
    hallucinations = []
    for i, entry in enumerate(eval_results):
        if i < len(pipeline_outputs):
            actual = entry.get("actual_output", "")
            if actual and "cannot find" not in actual.lower() and "not in" not in actual.lower():
                # This question might be unanswerable
                pipeline_entry = pipeline_outputs[i]
                expected = pipeline_entry.get("expected_answer", "")
                if expected and "i cannot" in expected.lower():
                    hallucinations.append({
                        "index": i,
                        "question": entry.get("question", ""),
                        "expected": expected[:100],
                        "actual": actual[:100],
                    })
    return hallucinations

hall_local = find_hallucinations(eval_local, pipeline_local)
hall_cloud = find_hallucinations(eval_cloud, pipeline_cloud)

print(f"Local: {len(hall_local)} potential hallucinations")
print(f"Cloud: {len(hall_cloud)} potential hallucinations")
if hall_local:
    print("\nExamples (Local):")
    for h in hall_local[:3]:
        print(f"  Q{h['index']}: {h['question'][:60]}...")
        print(f"    Expected: {h['expected'][:60]}...")
        print(f"    Got: {h['actual'][:60]}...")
print()

# Analysis insights
print("="*100)
print("KEY INSIGHTS & RECOMMENDATIONS")
print("="*100)
print()

faithfulness_gap = stats_cloud["Faithfulness"]["avg"] - stats_local["Faithfulness"]["avg"]
relevancy_gap = stats_cloud["Answer Relevancy"]["avg"] - stats_local["Answer Relevancy"]["avg"]
speed_diff = (timing_cloud['avg_generation_ms'] - timing_local['avg_generation_ms']) / timing_local['avg_generation_ms'] * 100 if timing_local['avg_generation_ms'] > 0 else 0

print(f"1. Quality Gap:")
if faithfulness_gap > 0.2:
    print(f"   ⚠ Cloud is {faithfulness_gap:.2%} better at faithfulness (staying in context)")
elif faithfulness_gap < -0.1:
    print(f"   ✓ Local is surprisingly good at faithfulness (only {abs(faithfulness_gap):.2%} worse)")
else:
    print(f"   ≈ Similarity gap in faithfulness: {faithfulness_gap:+.2%}")

print()
print(f"2. Speed Trade-off:")
if speed_diff < 0:
    print(f"   ✓ Local is {abs(speed_diff):.0f}% faster than cloud")
else:
    print(f"   ⚠ Cloud takes {speed_diff:.0f}% longer (network latency)")

print()
print(f"3. Cost-Quality Recommendation:")
if faithfulness_gap > 0.3:
    print(f"   → For production: Use CLOUD for critical applications (>30% quality improvement)")
    print(f"   → For draft/internal: Local model acceptable for cost savings")
elif faithfulness_gap < 0.1:
    print(f"   → LOCAL MODEL MAY BE SUFFICIENT - comparable quality at zero API cost")
else:
    print(f"   → HYBRID: Use local for most queries, cloud for high-accuracy needs")

print()
print("="*100)
