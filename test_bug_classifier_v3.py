"""
test_bug_classifier_v3.py
----------------------------
A larger, balanced OOD test: 24 examples (12 buggy/clean pairs) across 12
domains that appear NOWHERE in generate_bug_dataset_v3.py's function bank:
stacks, queues, circuit breakers, backoff, rate limiting, matrix transpose,
config merging, checksums, a different unit conversion, env var parsing,
run-length decoding, event dispatch.

Deliberately balanced 12/12 so a constant predictor scores exactly 50% --
any real result above that reflects genuine discrimination, not class
imbalance luck. Each pair is also tagged with the bug CATEGORY it mirrors
from generate_bug_dataset_v3.py's mutation types, so results can be broken
down the same way the CV script breaks down training performance --
letting you compare "does this category generalize" directly against
"does this category get learned" from cv_bug_classifier.py's table.

Run:
    python test_bug_classifier_v3.py
"""

import ast
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

ADAPTER_PATH = "./bug_classifier_lora_adapter"
BASE_MODEL = sys.argv[1] if len(sys.argv) > 1 else "microsoft/unixcoder-base"

# (code, label, category, reason)
NEW_EXAMPLES = [
    ("def pop_or_none(stack):\n    return stack.pop()",
     1, "remove_guard", "no guard against popping an empty stack -- crashes"),
    ("def pop_or_none(stack):\n    if not stack:\n        return None\n    return stack.pop()",
     0, "remove_guard", "guards against empty stack before popping"),

    ("def is_queue_full(queue, capacity):\n    return len(queue) > capacity",
     1, "flip_comparison", "off-by-one -- allows one item beyond capacity before reporting full"),
    ("def is_queue_full(queue, capacity):\n    return len(queue) >= capacity",
     0, "flip_comparison", "reports full exactly at capacity, not one past it"),

    ("def should_trip_breaker(failure_count, threshold):\n    return failure_count > threshold",
     1, "flip_comparison", "should trip AT the threshold, not one failure past it"),
    ("def should_trip_breaker(failure_count, threshold):\n    return failure_count >= threshold",
     0, "flip_comparison", "trips exactly when failures reach the threshold"),

    ("def backoff_seconds(attempt, base=2):\n    return base * attempt",
     1, "swap_arithmetic", "linear instead of exponential backoff -- defeats the purpose of backoff"),
    ("def backoff_seconds(attempt, base=2):\n    return base ** attempt",
     0, "swap_arithmetic", "correctly exponential"),

    ("def consume_token(tokens_remaining):\n    return tokens_remaining - 1",
     1, "remove_guard", "no guard against going negative -- silently breaks the rate limit"),
    ("def consume_token(tokens_remaining):\n    if tokens_remaining <= 0:\n        raise ValueError('no tokens remaining')\n    return tokens_remaining - 1",
     0, "remove_guard", "raises instead of silently going negative"),

    ("def transpose(matrix):\n    return [[row[i] for row in matrix] for i in range(len(matrix))]",
     1, "off_by_one", "uses row count instead of column count -- wrong for non-square matrices"),
    ("def transpose(matrix):\n    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]",
     0, "off_by_one", "correctly uses column count"),

    ("def get_setting(user_config, defaults, key):\n    if key in user_config or user_config[key] is not None:\n        return user_config[key]\n    return defaults.get(key)",
     1, "swap_boolop", "should be 'and' -- 'or' can raise KeyError or return an unintended value"),
    ("def get_setting(user_config, defaults, key):\n    if key in user_config and user_config[key] is not None:\n        return user_config[key]\n    return defaults.get(key)",
     0, "swap_boolop", "correctly requires both conditions"),

    ("def has_even_parity(bits):\n    return sum(bits) % 2 != 0",
     1, "flip_comparison", "function name says EVEN parity but the check is for odd"),
    ("def has_even_parity(bits):\n    return sum(bits) % 2 == 0",
     0, "flip_comparison", "correctly checks for even parity"),

    ("def hours_to_days(hours):\n    return hours / 12",
     1, "wrong_constant_scale", "wrong constant -- should divide by 24, not 12"),
    ("def hours_to_days(hours):\n    return hours / 24",
     0, "wrong_constant_scale", "correct conversion constant"),

    ("def get_port(env_value, default=8080):\n    return int(env_value)",
     1, "remove_guard", "no fallback or validation -- crashes on None or non-numeric input"),
    ("def get_port(env_value, default=8080):\n    if env_value is None or not env_value.isdigit():\n        return default\n    return int(env_value)",
     0, "remove_guard", "validates and falls back to default"),

    ("def decode_rle(pairs):\n    return ''.join(char * (count - 1) for char, count in pairs)",
     1, "off_by_one", "undercounts every run by one character"),
    ("def decode_rle(pairs):\n    return ''.join(char * count for char, count in pairs)",
     0, "off_by_one", "correctly repeats each character its full count"),

    ("def register_handler(handlers, event_name, handler):\n    if event_name not in handlers:\n        handlers[event_name] = []\n    return handlers[event_name]",
     1, "silent_wrong_default", "never actually appends the handler -- silently does nothing"),
    ("def register_handler(handlers, event_name, handler):\n    if event_name not in handlers:\n        handlers[event_name] = []\n    handlers[event_name].append(handler)",
     0, "silent_wrong_default", "correctly appends the handler"),
]


def main():
    print(f"Loading base model + LoRA adapter from {ADAPTER_PATH} (base: {BASE_MODEL})...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    results = []
    print(f"\nTesting on {len(NEW_EXAMPLES)} FRESH hand-written examples "
          f"across 12 new domains (balanced 12 buggy / 12 clean):\n")

    for code, expected_label, category, reason in NEW_EXAMPLES:
        code_norm = ast.unparse(ast.parse(code))
        inputs = tokenizer(code_norm, truncation=True, padding="max_length",
                            max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            predicted_label = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_label].item()

        is_correct = predicted_label == expected_label
        results.append((category, is_correct))
        label_name = {0: "clean", 1: "buggy"}
        mark = "PASS" if is_correct else "FAIL"

        print(f"[{mark}] predicted={label_name[predicted_label]} "
              f"(confidence {confidence:.2f})  expected={label_name[expected_label]}  [{category}]")
        print(f"       {reason}")
        print()

    correct = sum(1 for _, ok in results if ok)
    accuracy = correct / len(results)
    print(f"Overall: {correct}/{len(results)} ({accuracy:.1%})  -- constant predictor floor = 50.0%")

    print("\nBy category (compare against cv_bug_classifier.py's training-side breakdown):")
    from collections import defaultdict
    by_cat = defaultdict(lambda: [0, 0])
    for category, ok in results:
        by_cat[category][0] += int(ok)
        by_cat[category][1] += 1
    for cat, (c, t) in sorted(by_cat.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        print(f"  {cat:22s} {c}/{t} ({c/t:.0%})")


if __name__ == "__main__":
    main()