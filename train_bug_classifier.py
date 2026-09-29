"""
train_bug_classifier.py
-------------------------
Fine-tunes DistilBERT with a LoRA adapter to classify a code snippet as
"looks risky/buggy" (1) or "looks clean" (0).

Includes several honesty checks:
  1. GROUP-AWARE train/test split (no leakage from near-duplicate mutants)
  2. Majority-class baseline (printed + saved, so raw accuracy is never
     reported without context)
  3. Per-epoch trajectory logging (to catch under/overfitting, not just
     the final number)
  4. BALANCED accuracy for best-checkpoint selection -- raw accuracy and
     F1 can both be maximized by a trivial constant predictor when the
     dataset is class-imbalanced (this is exactly what happened on the
     v2 dataset: a "recall=1.0 always" predictor scored the highest F1
     AND highest accuracy of the entire run). Balanced accuracy -- the
     average of per-class recall -- scores a constant predictor at
     exactly 0.5 regardless of class imbalance, so it can't be fooled
     the same way.

"""

import sys
import json
import numpy as np
from datasets import Dataset
from sklearn.model_selection import GroupShuffleSplit
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score,
)
import torch

# MODEL_NAME = "distilbert-base-uncased"
MODEL_NAME = "microsoft/unixcoder-base"
SEED = 42

# Seeds Python's random, NumPy, and PyTorch (CPU + CUDA/MPS) together --
# without this, LoRA's random adapter initialization, dropout masks, and
# batch shuffling all vary run-to-run even on identical data, making it
# impossible to tell a real improvement from a lucky random draw.
set_seed(SEED)


def load_data(path):
    return [json.loads(line) for line in open(path)]


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
    }

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.cross_entropy(
            outputs.logits, labels,
            weight=self.class_weights.to(outputs.logits.device),
        )
        return (loss, outputs) if return_outputs else loss

def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else "bug_dataset_v2.jsonl"

    print(f"Loading dataset from {data_path}...")
    examples = load_data(data_path)

    groups = [ex["source_id"] for ex in examples]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(examples, groups=groups))

    train_examples = [examples[i] for i in train_idx]
    test_examples = [examples[i] for i in test_idx]

    train_groups = {examples[i]["source_id"] for i in train_idx}
    test_groups = {examples[i]["source_id"] for i in test_idx}
    overlap = train_groups & test_groups
    assert not overlap, f"Leakage detected: source_ids {overlap} appear in both splits"
    print(f"Verified no group overlap between train ({len(train_groups)} source functions) "
          f"and test ({len(test_groups)} source functions)")

    test_labels = [ex["label"] for ex in test_examples]
    majority_class = max(set(test_labels), key=test_labels.count)
    majority_baseline_acc = test_labels.count(majority_class) / len(test_labels)
    print(f"Test set: {test_labels.count(0)} clean, {test_labels.count(1)} buggy")
    print(f"Majority-class baseline accuracy (predicting {majority_class} always): "
          f"{majority_baseline_acc:.3f}")
    print("Note: a constant predictor always scores exactly 0.500 balanced_accuracy, "
          "regardless of class imbalance -- that's the number to trust most.\n")

    train_labels = [ex["label"] for ex in train_examples]
    print(f"Train set: {train_labels.count(0)} clean, {train_labels.count(1)} buggy "
          f"({train_labels.count(1) / len(train_labels):.1%} buggy)\n")

    dataset = {
        "train": Dataset.from_list(train_examples),
        "test": Dataset.from_list(test_examples),
    }
    print(f"Train: {len(dataset['train'])}, Eval: {len(dataset['test'])}")

    print(f"Loading tokenizer and base model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["code"], truncation=True, padding="max_length", max_length=128)

    dataset = {split: ds.map(tokenize, batched=True) for split, ds in dataset.items()}
    for split, ds in dataset.items():
        ds = ds.rename_column("label", "labels")
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        dataset[split] = ds

    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        # target_modules=["q_lin", "v_lin"],
        target_modules=["query", "value"],
    )
    model = get_peft_model(base_model, lora_config)

    print("\nTrainable parameters (this is the point of LoRA):")
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="./bug_classifier_output",
        num_train_epochs=15,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_balanced_accuracy",
        greater_is_better=True,
        logging_steps=5,
        # learning_rate=1e-3,
        learning_rate=5e-4,
        report_to="none",
        use_cpu=True,  # MPS (Apple Silicon GPU) has known non-deterministic
                       # kernels even with a seed set -- forcing CPU trades a
                       # little speed for TRUE reproducibility, worthwhile on
                       # a dataset this small (208 examples finishes in
                       # seconds either way).
    )

    n0, n1 = train_labels.count(0), train_labels.count(1)
    class_weights = torch.tensor(
        [len(train_labels) / (2 * n0), len(train_labels) / (2 * n1)],
        dtype=torch.float,
    )
    print(f"Class weights: clean={class_weights[0]:.2f}, buggy={class_weights[1]:.2f}")

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=dataset["train"],
    #     eval_dataset=dataset["test"],
    #     compute_metrics=compute_metrics,
    # )

    trainer =WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
    )

    print("\nTraining...")
    trainer.train()

    print("\nPer-epoch eval history:")
    for entry in trainer.state.log_history:
        if "eval_accuracy" in entry:
            print(f"  epoch {entry['epoch']:.1f}: "
                  f"acc={entry['eval_accuracy']:.3f} "
                  f"bal_acc={entry['eval_balanced_accuracy']:.3f} "
                  f"f1={entry['eval_f1']:.3f} "
                  f"precision={entry['eval_precision']:.3f} "
                  f"recall={entry['eval_recall']:.3f} "
                  f"loss={entry['eval_loss']:.3f}")

    print("\nFinal evaluation on held-out test set (best checkpoint by balanced_accuracy):")
    metrics = trainer.evaluate()
    from collections import defaultdict
    preds = np.argmax(trainer.predict(dataset["test"]).predictions, axis=-1)
    by_type = defaultdict(lambda: [0, 0])  # [correct, total]
    for ex, pred in zip(test_examples, preds):
        by_type[ex["mutation"]][0] += int(pred == ex["label"])
        by_type[ex["mutation"]][1] += 1
    print("\nAccuracy by mutation type:")
    for mtype, (correct, total) in sorted(by_type.items()):
        print(f"  {mtype:22s} {correct}/{total}")
    # for key, value in metrics.items():
    #     print(f"  {key}: {value}")



    print(f"\nFor comparison, majority-class baseline accuracy was: {majority_baseline_acc:.3f}")
    print("A constant predictor scores exactly 0.500 balanced_accuracy -- "
          f"this model scored {metrics['eval_balanced_accuracy']:.3f}")
    diff = metrics["eval_balanced_accuracy"] - 0.5
    print(f"Balanced accuracy above trivial baseline: {diff:+.3f}")
    if diff < 0.05:
        print("WARNING: barely above the trivial 0.5 balanced-accuracy floor. "
              "Likely still not learning a meaningful signal.")

    model.save_pretrained("./bug_classifier_lora_adapter")
    tokenizer.save_pretrained("./bug_classifier_lora_adapter")
    print("\nSaved LoRA adapter to ./bug_classifier_lora_adapter")

    with open("training_metrics.json", "w") as f:
        json.dump({
            **{k: float(v) for k, v in metrics.items()},
            "majority_baseline_accuracy": majority_baseline_acc,
        }, f, indent=2)
    print("Metrics saved to training_metrics.json")


if __name__ == "__main__":
    main()