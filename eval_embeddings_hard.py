"""
eval_embeddings_hard.py
------------------------
A harder, more discriminating retrieval eval than eval_embeddings.py.
Two upgrades:
  1. Top-1 accuracy (must rank the correct chunk FIRST, not just top-3)
     and Mean Reciprocal Rank (MRR) -- a more granular metric than pass/fail.
  2. A test set specifically designed to be hard: queries phrased far from
     the actual function/docstring wording, plus near-duplicate confusable
     functions (reverse_string vs reverse_list, divide vs safe_divide,
     deduplicate vs find_duplicates) that force real semantic discrimination,
     not just "is this vaguely related or not."

Run:
    python eval_embeddings_hard.py
"""

import sys
sys.path.insert(0, "src")

from retriever import CodebaseRetriever

REPO_PATH = "sample_repo"

# Harder test set. Query phrasing deliberately avoids reusing words from the
# function name or docstring where possible, to test real semantic
# understanding rather than near-literal keyword overlap.
TEST_SET = [
    # Near-duplicate pairs -- the model must distinguish, not just find
    # "something about reversing" or "something about division"
    ("flip the order of characters in a word",       "string_utils.py::reverse_string"),
    ("flip the order of items in a collection",       "string_utils.py::reverse_list"),
    ("divide two numbers without ever crashing",      "string_utils.py::safe_divide"),
    ("divide two numbers, raising an error on zero",  "calculator.py::divide"),
    ("get the items that occur more than once",       "string_utils.py::find_duplicates"),
    ("remove repeated items but keep the first occurrence", "string_utils.py::deduplicate"),

    # Indirect phrasing -- avoids the function's own vocabulary entirely
    ("check if a word is spelled the same in both directions", "string_utils.py::is_palindrome"),
    ("keep a number from going above or below certain bounds", "string_utils.py::clamp"),
    ("turn a list of lists into one single list",     "string_utils.py::flatten"),
    ("break a big list into smaller equal-sized groups", "string_utils.py::chunk_list"),
    ("clean up extra spaces in a sentence",            "string_utils.py::normalize_whitespace"),

    # Behavioral/edge-case phrasing -- requires understanding what the code
    # DOES under specific conditions, not just its general topic
    ("what happens if you call average on an empty list", "calculator.py::average"),
    ("which function returns a fallback value instead of erroring", "string_utils.py::safe_divide"),
]


def evaluate(embedding_model: str) -> dict:
    print(f"\nBuilding index with: {embedding_model}")
    retriever = CodebaseRetriever(embedding_model=embedding_model)
    n_chunks = retriever.build_index(REPO_PATH)
    print(f"Indexed {n_chunks} total chunks (bigger haystack = harder task)")

    top1_hits = 0
    reciprocal_ranks = []
    log = []

    for query, expected_id in TEST_SET:
        # Ask for more results than we need so we can compute MRR even if
        # the correct answer lands outside top-1.
        results = retriever.search(query, k=5)
        retrieved_ids = [f"{r['file']}::{r['name']}" for r in results]

        top1_hit = retrieved_ids[0] == expected_id if retrieved_ids else False
        top1_hits += top1_hit

        if expected_id in retrieved_ids:
            rank = retrieved_ids.index(expected_id) + 1  # 1-indexed
            reciprocal_ranks.append(1 / rank)
        else:
            rank = None
            reciprocal_ranks.append(0)

        log.append((query, expected_id, retrieved_ids, top1_hit, rank))

    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return {
        "model": embedding_model,
        "top1_accuracy": top1_hits / len(TEST_SET),
        "top1_hits": top1_hits,
        "mrr": mrr,
        "total": len(TEST_SET),
        "log": log,
    }


def print_report(result: dict):
    print(f"\n{'=' * 65}")
    print(f"Model: {result['model']}")
    print(f"Top-1 accuracy: {result['top1_hits']}/{result['total']} "
          f"({result['top1_accuracy']*100:.1f}%)")
    print(f"MRR: {result['mrr']:.3f}  (1.0 = always ranked correct answer #1)")
    print(f"{'=' * 65}")
    for query, expected, retrieved, hit, rank in result["log"]:
        rank_str = f"rank {rank}" if rank else "NOT FOUND in top-5"
        mark = "PASS" if hit else "FAIL"
        print(f"  [{mark}] \"{query}\"  ({rank_str})")
        print(f"         expected: {expected}")
        if not hit:
            print(f"         top result was: {retrieved[0] if retrieved else 'none'}")


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

    print(f"\n{'=' * 65}")
    print("SUMMARY")
    print(f"{'=' * 65}")
    for r in all_results:
        print(f"  {r['model']:55s} top1={r['top1_accuracy']*100:5.1f}%  MRR={r['mrr']:.3f}")