"""
test_bug_classifier.py
------------------------
Loads the trained LoRA adapter and runs it against BRAND NEW, hand-written
functions -- none of these came from generate_bug_dataset.py's
CLEAN_FUNCTIONS bank or its mutation process. This is the real
generalization check: everything you've tested so far (train/test split,
majority baseline, overfitting curve) came from the same mutation-generated
dataset. This is the first check using genuinely independent examples.

Run:
    python test_bug_classifier.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

ADAPTER_PATH = "./bug_classifier_lora_adapter"
BASE_MODEL = "distilbert-base-uncased"

# Hand-written, genuinely new examples. Different domain (date/string
# parsing) from the mutation bank's math/list utilities on purpose --
# a stronger test than more of the same kind of function.
NEW_EXAMPLES = [
    # (code, expected_label, why)
    (
        "def parse_date(s):\n    parts = s.split('-')\n    return int(parts[0]), int(parts[1]), int(parts[2])",
        1,  # buggy -- no check that split produced exactly 3 parts, will
            # crash with IndexError on malformed input
        "no validation that input has the expected format",
    ),
    (
        "def parse_date(s):\n    parts = s.split('-')\n    if len(parts) != 3:\n        raise ValueError('invalid date format')\n    return int(parts[0]), int(parts[1]), int(parts[2])",
        0,  # clean -- validates before using
        "validates format before parsing",
    ),
    (
        "def get_extension(filename):\n    return filename.split('.')[-1]",
        1,  # buggy -- returns the whole filename if there's no '.' at all,
            # silently wrong rather than crashing
        "silently wrong on filenames with no extension",
    ),
    (
        "def get_extension(filename):\n    if '.' not in filename:\n        return ''\n    return filename.split('.')[-1]",
        0,
        "explicitly handles the no-extension case",
    ),
    (
        "def calculate_discount(price, percent):\n    return price - (price * percent)",
        1,  # buggy -- percent isn't divided by 100, so passing 20 (meaning
            # "20%") would subtract 20x the price, not 20% of it
        "percent not divided by 100 -- a real, subtle logic bug",
    ),
    (
        "def calculate_discount(price, percent):\n    return price - (price * percent / 100)",
        0,
        "correctly converts percent to a fraction",
    ),
    (
        "def get_middle_element(items):\n    return items[len(items) // 2]",
        1,  # buggy -- crashes on empty list, no guard
        "no empty-list guard",
    ),
    (
        "def get_middle_element(items):\n    if not items:\n        raise ValueError('empty list')\n    return items[len(items) // 2]",
        0,
        "guards against empty input",
    ),
]


def main():
    print(f"Loading base model + LoRA adapter from {ADAPTER_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=2)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    correct = 0
    print(f"\nTesting on {len(NEW_EXAMPLES)} brand-new, hand-written examples "
          f"(none seen during training):\n")

    for code, expected_label, reason in NEW_EXAMPLES:
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
        print(f"       code: {code.splitlines()[0]}...")
        print()

    accuracy = correct / len(NEW_EXAMPLES)
    print(f"Accuracy on brand-new hand-written examples: {correct}/{len(NEW_EXAMPLES)} "
          f"({accuracy:.1%})")
    print("\nThis number matters more than your training-eval accuracy -- these")
    print("examples share NOTHING with the mutation-generated training data")
    print("except the general concept of 'a Python function with a bug'.")


if __name__ == "__main__":
    main()