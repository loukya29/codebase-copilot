"""
train_bug_classifier.py
-------------------------
Fine-tunes DistilBERT with a LoRA adapter to classify a code snippet as
"looks risky/buggy" (1) or "looks clean" (0). Small dataset (~150 examples),
LoRA keeps training fast and cheap even on CPU.

Run:
    python train_bug_classifier.py bug_dataset.jsonl
"""

import sys
import json
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

MODEL_NAME = "distilbert-base-uncased"


def load_data(path):
    examples = [json.loads(line) for line in open(path)]
    return Dataset.from_list(examples)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
    }


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else "bug_dataset.jsonl"

    print(f"Loading dataset from {data_path}...")
    dataset = load_data(data_path)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    print(f"Train: {len(dataset['train'])}, Eval: {len(dataset['test'])}")

    print(f"Loading tokenizer and base model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["code"], truncation=True, padding="max_length", max_length=128)

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    # LoRA config -- this is the "parameter-efficient" part. Instead of
    # updating all ~66M DistilBERT parameters, we freeze the base model and
    # only train small low-rank adapter matrices injected into the attention
    # layers (q_lin, v_lin -- DistilBERT's query/value projection names).
    # This is why training a "toy" classifier on CPU is actually feasible.
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,                    # rank of the adapter matrices -- small and cheap
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],
    )
    model = get_peft_model(base_model, lora_config)

    print("\nTrainable parameters (this is the point of LoRA):")
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir="./bug_classifier_output",
        num_train_epochs=8,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=5,
        learning_rate=2e-4,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
    )

    print("\nTraining...")
    trainer.train()

    print("\nFinal evaluation on held-out test set:")
    metrics = trainer.evaluate()
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Save the LoRA adapter -- small, just the adapter weights, not the
    # full base model (that's the whole point of PEFT: a few MB, not ~260MB)
    model.save_pretrained("./bug_classifier_lora_adapter")
    tokenizer.save_pretrained("./bug_classifier_lora_adapter")
    print("\nSaved LoRA adapter to ./bug_classifier_lora_adapter")

    # Write metrics to a file too, so you have a real, timestamped record
    # to reference later -- not just terminal output you'll lose
    with open("training_metrics.json", "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    print("Metrics saved to training_metrics.json")


if __name__ == "__main__":
    main()