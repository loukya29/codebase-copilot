"""
test_bug_classifier_v2.py
----------------------------
A SECOND, genuinely fresh out-of-distribution test. The original 8
hand-written examples covered patterns (missing validation, wrong unit
scaling) that v2's training data now explicitly includes -- reusing them
would be a softer, less honest test of generalization. These 8 are new
domains entirely: caching, pagination, password validation, and sorting --
none represented in generate_bug_dataset_v2.py's function bank at all.

Run:
    python test_bug_classifier_v2.py
"""
import ast

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

ADAPTER_PATH = "./bug_classifier_lora_adapter"
# BASE_MODEL = "distilbert-base-uncased"
BASE_MODEL = "microsoft/unixcoder-base"

NEW_EXAMPLES = [
    (
        "def get_cached(cache, key, compute_fn):\n    if key in cache:\n        return cache[key]\n    value = compute_fn()\n    return value",
        1,  # buggy -- computes and returns the value but never stores it
            # in cache, defeating the entire purpose of caching
        "computed value never actually gets cached",
    ),
    (
        "def get_cached(cache, key, compute_fn):\n    if key in cache:\n        return cache[key]\n    value = compute_fn()\n    cache[key] = value\n    return value",
        0,
        "correctly stores the computed value before returning",
    ),
    (
        "def paginate(items, page, page_size):\n    start = page * page_size\n    return items[start:start + page_size]",
        1,  # buggy -- page 1 (first page, if 1-indexed) would skip the
            # first page_size items; off-by-one in the common convention
            # where pages are 1-indexed, not 0-indexed
        "assumes 0-indexed pages without validating or documenting it -- a common off-by-one source",
    ),
    (
        "def paginate(items, page, page_size):\n    if page < 1:\n        raise ValueError('page must be >= 1')\n    start = (page - 1) * page_size\n    return items[start:start + page_size]",
        0,
        "explicit 1-indexed convention with validation",
    ),
    (
        "def is_strong_password(s):\n    return len(s) >= 8",
        1,  # buggy -- only checks length, ignores the "strong" criteria
            # a function named this way implies (mixed case, digits, etc.)
        "name implies more validation than the implementation actually does",
    ),
    (
        "def is_strong_password(s):\n    if len(s) < 8:\n        return False\n    has_upper = any(c.isupper() for c in s)\n    has_digit = any(c.isdigit() for c in s)\n    return has_upper and has_digit",
        0,
        "actually checks multiple strength criteria",
    ),
    (
        "def sort_by_key(items, key_fn):\n    return sorted(items, key=key_fn, reverse=True)",
        1,  # buggy -- hardcodes reverse=True with no way to control sort
            # direction, despite no name/doc indicating descending-only
        "silently hardcodes descending order with no way to control it",
    ),
    (
        "def sort_by_key(items, key_fn, descending=False):\n    return sorted(items, key=key_fn, reverse=descending)",
        0,
        "sort direction is an explicit, controllable parameter",
    ),
]


def main():
    print(f"Loading base model + LoRA adapter from {ADAPTER_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    correct = 0
    print(f"\nTesting on {len(NEW_EXAMPLES)} FRESH hand-written examples "
          f"(caching, pagination, validation, sorting -- none of these "
          f"domains appear in the v2 training data):\n")

    for code, expected_label, reason in NEW_EXAMPLES:
        code = ast.unparse(ast.parse(code))  # match training-data formatting
        inputs = tokenizer(code, truncation=True, padding="max_length",
                           max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            predicted_label = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_label].item()

        is_correct = predicted_label == expected_label
        correct += is_correct
        label_name = {0: "clean", 1: "buggy"}
        mark = "PASS" if is_correct else "FAIL"

        print(f"[{mark}] predicted={label_name[predicted_label]} "
              f"(confidence {confidence:.2f})  expected={label_name[expected_label]}")
        print(f"       reason: {reason}")
        print()

    accuracy = correct / len(NEW_EXAMPLES)
    print(f"Accuracy on fresh OOD examples: {correct}/{len(NEW_EXAMPLES)} ({accuracy:.1%})")
    print("Compare against the v1 result (4/8, 50% -- pure constant-predictor behavior)")


if __name__ == "__main__":
    main()