"""
eval_embeddings.py
-------------------
Compares two Hugging Face embedding models on retrieval accuracy, using a
small hand-labeled set of (query, expected_chunk_id) pairs.

Metric: top-k accuracy -- for each query, did the expected chunk appear
ANYWHERE in the top-k results? This is a standard, simple IR metric and
an easy one to explain and defend in an interview.

Run:
    python eval_embeddings.py
"""

import sys
sys.path.insert(0, "src")

from retriever import CodebaseRetriever

sys.path.insert(0, "src")


REPO_PATH = "sample_repo"

# Hand-labeled test set: (natural-language query, expected chunk id).
# Expand this yourself as you add more functions -- 15-20 examples is a
# reasonable size to defend in an interview ("I hand-labeled 18 queries").
TEST_SET = [
    ("where is division handled",              "calculator.py::divide"),
    ("how do I add two numbers",                "calculator.py::add"),
    ("subtract one number from another",        "calculator.py::subtract"),
    ("multiply two values",                     "calculator.py::multiply"),
    ("compute the average of a list",           "calculator.py::average"),
    ("what happens when dividing by zero",      "calculator.py::divide"),
    ("mean of a list of numbers",               "calculator.py::average"),
    ("product of two numbers",                  "calculator.py::multiply"),
    ("sum two values together",                 "calculator.py::add"),
    ("difference between two numbers",          "calculator.py::subtract"),
]


def evaluate(embedding_model: str, k: int = 3) -> dict:
    print(f"\nBuilding index with: {embedding_model}")
    retriever = CodebaseRetriever(embedding_model=embedding_model)
    retriever.build_index(REPO_PATH)

    hits = 0
    results_log = []

    for query, expected_id in TEST_SET:
        results = retriever.search(query, k=k)
        retrieved_ids = [f"{r['file']}::{r['name']}" for r in results]
        hit = expected_id in retrieved_ids
        hits += hit
        results_log.append((query, expected_id, retrieved_ids, hit))

    accuracy = hits / len(TEST_SET)
    return {"model": embedding_model, "accuracy": accuracy, "hits": hits,
            "total": len(TEST_SET), "log": results_log}


def print_report(result: dict):
    print(f"\n{'=' * 60}")
    print(f"Model: {result['model']}")
    print(f"Top-{3} accuracy: {result['hits']}/{result['total']} "
          f"({result['accuracy']*100:.1f}%)")
    print(f"{'=' * 60}")
    for query, expected, retrieved, hit in result["log"]:
        mark = "PASS" if hit else "FAIL"
        print(f"  [{mark}] \"{query}\"")
        print(f"         expected: {expected}")
        if not hit:
            print(f"         got:      {retrieved}")


if __name__ == "__main__":
    models_to_compare = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "flax-sentence-embeddings/st-codesearch-distilroberta-base",
    ]

    all_results = []
    for model_name in models_to_compare:
        result = evaluate(model_name)
        all_results.append(result)
        print_report(result)

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for r in all_results:
        print(f"  {r['model']:50s} {r['accuracy']*100:.1f}%")