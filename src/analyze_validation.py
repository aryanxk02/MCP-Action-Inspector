"""List incorrectly predicted examples from the validation dataset."""

import csv
import json
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABELS = ["Read", "Write", "Execute", "Destructive", "Financial", "Other"]

VALIDATION_FILE = "data/validation.jsonl"
MODEL_DIR = "src/slm_model"
OUTPUT_FILE = "src/validation_errors.csv"

BATCH_SIZE = 16
MAX_LENGTH = 256


def load_jsonl(path):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def make_text(row):
    return "\n".join([
        f"name: {row.get('name') or ''}",
        f"description: {row.get('description') or ''}",
        f"input_schema: {row.get('input_schema') or ''}",
    ])


def main():
    # Use the trained model saved by main.py.
    model_dir = MODEL_DIR

    if not os.path.exists(os.path.join(model_dir, "config.json")):
        raise FileNotFoundError(
            f"Could not find trained model at '{model_dir}'. "
            "Run main.py first to train the model."
        )

    print("Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # Force CPU.
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    print("Loading validation data...")

    rows = load_jsonl(VALIDATION_FILE)

    errors = []

    print(f"Checking {len(rows)} validation examples...")

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]

        texts = [make_text(row) for row in batch]

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        # DistilBERT does not use token_type_ids.
        inputs.pop("token_type_ids", None)

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        # No gradients needed during prediction.
        with torch.inference_mode():
            logits = model(**inputs).logits
            predicted_ids = logits.argmax(dim=-1).tolist()

        for row, predicted_id in zip(batch, predicted_ids):
            actual = row["category"]
            predicted = LABELS[predicted_id]

            if actual != predicted:
                errors.append({
                    "record_id": row["record_id"],
                    "name": row.get("name", ""),
                    "description": row.get("description", ""),
                    "actual": actual,
                    "predicted": predicted,
                })

    # Print errors to the terminal.
    print()
    print("=" * 80)
    print(f"INCORRECT PREDICTIONS: {len(errors)}")
    print("=" * 80)

    for i, error in enumerate(errors, start=1):
        print()
        print(f"[{i}] Record ID: {error['record_id']}")
        print(f"Name:       {error['name']}")
        print(f"Actual:     {error['actual']}")
        print(f"Predicted:  {error['predicted']}")
        print(f"Description: {error['description']}")
        print("-" * 80)

    # Save errors to a CSV so they are easy to inspect.
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "record_id",
                "name",
                "description",
                "actual",
                "predicted",
            ],
        )

        writer.writeheader()
        writer.writerows(errors)

    print()
    print("=" * 80)
    print(f"Total validation examples: {len(rows)}")
    print(f"Incorrect predictions:     {len(errors)}")
    print(f"Validation accuracy:       {(len(rows) - len(errors)) / len(rows):.4f}")
    print(f"Saved errors to:            {OUTPUT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
