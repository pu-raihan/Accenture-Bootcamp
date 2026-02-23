#!/usr/bin/env python3
from retrieve import retrieve_chunks
from generate import generate_answer


def answer_question(query: str):
    retrieval_result = retrieve_chunks(query)
    generation_result = generate_answer(query=query, chunks=retrieval_result.get("retrieved_chunks", []))
    return {**generation_result, "retrieval": retrieval_result}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    args = parser.parse_args()
    result = answer_question(args.query)
    import json

    print(json.dumps(result, indent=2, ensure_ascii=False))
