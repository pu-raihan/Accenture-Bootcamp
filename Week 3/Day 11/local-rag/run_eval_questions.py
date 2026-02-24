import sys
import os
import json
from datetime import datetime

from pipeline import answer_question

def run_eval_questions(output_file="pipeline_outputs_local.json"):
    
    print(f"\n{'-'*60}")
    print(f"Running pipeline evaluation...")
    print(f"Loading test questions from: test_questions.json")
    print(f"Output file: {output_file}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"{'-'*60}\n")
    
    with open("test_questions.json") as f:
        test_questions = json.load(f)
    
    print(f"Loaded {len(test_questions)} test questions")
    
    results = []
    failed_count = 0
    
    for i, q in enumerate(test_questions, 1):
        try:
            print(f"[{i}/{len(test_questions)}] Processing: {q['question'][:60]}...")
            
            output = answer_question(q["question"])
            
            retrieved_context = []
            if output.get("retrieval") and output["retrieval"].get("retrieved_chunks"):
                retrieved_context = [c["content"] for c in output["retrieval"]["retrieved_chunks"]]
            
            result = {
                "id": q["id"],
                "question": q["question"],
                "expected_answer": q["expected_answer"],
                "actual_answer": output.get("answer", ""),
                "retrieved_context": retrieved_context,
                "model": output.get("model", "unknown"),
                "retrieval_time_ms": output.get("retrieval", {}).get("retrieval_time_ms", 0),
                "generation_time_ms": output.get("generation_time_ms", 0),
            }
            
            results.append(result)
            print(f"Success (retrieval: {len(retrieved_context)} chunks, gen: {result['generation_time_ms']}ms)")
            
        except Exception as e:
            print(f"Failed: {str(e)}")
            failed_count += 1
            results.append({
                "id": q["id"],
                "question": q["question"],
                "expected_answer": q["expected_answer"],
                "actual_answer": f"ERROR: {str(e)}",
                "retrieved_context": [],
                "model": "error",
                "retrieval_time_ms": 0,
                "generation_time_ms": 0,
                "error": str(e)
            })
    
    output_path = os.path.join(os.path.dirname(__file__), output_file)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'-'*60}")
    print(f"Evaluation Complete!")
    print(f"  Total questions: {len(test_questions)}")
    print(f"  Successful: {len(test_questions) - failed_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Output saved to: {output_path}")
    print(f"{'-'*60}\n")
    
    return results

if __name__ == "__main__":
    # Allow specifying output file as command-line argument
    output_file = sys.argv[1] if len(sys.argv) > 1 else "pipeline_outputs_local.json"
    run_eval_questions(output_file)
