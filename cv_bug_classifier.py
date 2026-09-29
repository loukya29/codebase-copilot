"""
cv_bug_classifier.py
----------------------
5-fold GROUP cross-validation for the bug classifier.

Why this exists: a single 80/20 split leaves ~46 test examples (~5 per
mutation type), so every metric -- overall and per type -- is dominated by
noise. Here every source function lands in the test fold exactly once, so
per-type accuracy is measured on the WHOLE dataset (~40-60 examples per type).

It also trains for a fixed number of epochs with NO best-checkpoint
selection. Picking the best epoch by test-set score (load_best_model_at_end
in train_bug_classifier.py) is optimistic, because the choice itself uses
the test data. Fixed epochs gives an unbiased number.

Reuses MODEL_NAME, SEED, load_data and WeightedTrainer from
train_bug_classifier.py, so both scripts stay in sync.

Run:
    python cv_bug_classifier.py bug_dataset_v2.jsonl
"""

import json
import sys
from collections import defaultdict

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)

from train_bug_classifier import MODEL_NAME, SEED, WeightedTrainer, load_data

N_FOLDS = 5
NUM_EPOCHS = 15


def to_dataset(examples, tokenizer):
    ds = Dataset.from_list([{"code": e["code"], "label": e["label"]} for e in examples])
    ds = ds.map(
        lambda b: tokenizer(b["code"], truncation=True, padding="max_length", max_length=128),
        batched=True,
    )
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "bug_dataset_v2.jsonl"
    examples = load_data(path)
    if "mutation" not in examples[0]:
        sys.exit("Dataset has no 'mutation' field -- regenerate it with the updated generator.")

    groups = [e["source_id"] for e in examples]
    all_labels = [e["label"] for e in examples]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    oof_preds = np.full(len(examples), -1)
    fold_scores = []
    splitter = GroupKFold(n_splits=N_FOLDS)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(examples, groups=groups)):
        set_seed(SEED)  # same init for every fold, so folds differ only by data
        train_ex = [examples[i] for i in train_idx]
        test_ex = [examples[i] for i in test_idx]

        # Sanity check: no source function in both train and test
        assert not ({e["source_id"] for e in train_ex} & {e["source_id"] for e in test_ex})

        labels = [e["label"] for e in train_ex]
        n0, n1 = labels.count(0), labels.count(1)
        class_weights = torch.tensor(
            [len(labels) / (2 * n0), len(labels) / (2 * n1)], dtype=torch.float
        )

        base = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        model = get_peft_model(base, LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "value"],
        ))

        args = TrainingArguments(
            output_dir=f"./cv_output/fold{fold}",
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            learning_rate=5e-4,
            eval_strategy="no",     # no peeking at the test fold during training
            save_strategy="no",
            logging_strategy="no",
            report_to="none",
            use_cpu=True,
            seed=SEED,
        )

        trainer = WeightedTrainer(
            model=model,
            args=args,
            train_dataset=to_dataset(train_ex, tokenizer),
            class_weights=class_weights,
        )
        print(f"\n=== Fold {fold + 1}/{N_FOLDS}: training on {len(train_ex)}, "
              f"testing on {len(test_ex)} ===")
        trainer.train()

        preds = np.argmax(trainer.predict(to_dataset(test_ex, tokenizer)).predictions, axis=-1)
        oof_preds[test_idx] = preds
        score = balanced_accuracy_score([e["label"] for e in test_ex], preds)
        fold_scores.append(score)
        print(f"Fold {fold + 1} balanced accuracy: {score:.3f}")

    assert (oof_preds >= 0).all(), "some examples were never tested"

    print("\n" + "=" * 60)
    print(f"Per-fold balanced accuracy: {', '.join(f'{s:.3f}' for s in fold_scores)}")
    print(f"Mean {np.mean(fold_scores):.3f}  std {np.std(fold_scores):.3f}")
    print(f"Pooled out-of-fold balanced accuracy: "
          f"{balanced_accuracy_score(all_labels, oof_preds):.3f}  (constant predictor = 0.500)")

    by_type = defaultdict(lambda: [0, 0])  # [correct, total]
    for e, p in zip(examples, oof_preds):
        by_type[e["mutation"]][0] += int(p == e["label"])
        by_type[e["mutation"]][1] += 1

    print("\nOut-of-fold accuracy by mutation type (every example tested once, worst first):")
    for mtype, (correct, total) in sorted(by_type.items(), key=lambda kv: kv[1][0] / kv[1][1]):
        print(f"  {mtype:22s} {correct:3d}/{total:<3d} ({correct / total:.0%})")

    with open("cv_predictions.jsonl", "w") as f:
        for e, p in zip(examples, oof_preds):
            f.write(json.dumps({**e, "predicted": int(p), "correct": int(p) == e["label"]}) + "\n")
    print("\nEvery example's prediction saved to cv_predictions.jsonl")


if __name__ == "__main__":
    main()