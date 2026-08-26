"""Find validation errors and summarize the three most common failure modes."""

import json
import os
from collections import Counter, defaultdict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABELS = ["Read", "Write", "Execute", "Destructive", "Financial", "Other"]
MODEL_DIR = "slm_model"
BATCH_SIZE = 32


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
    model_dir = MODEL_DIR
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        model_dir = "slm_model/slm_model"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    rows = load_jsonl("data/validation.jsonl")
    errors = []

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        inputs = tokenizer(
            [make_text(row) for row in batch],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        with torch.no_grad():
            predictions = model(**inputs).logits.argmax(dim=-1).tolist()

        for row, prediction_id in zip(batch, predictions):
            actual = row["category"]
            predicted = LABELS[prediction_id]
            if actual != predicted:
                errors.append({
                    "record_id": row["record_id"],
                    "name": row.get("name", ""),
                    "description": row.get("description", ""),
                    "actual": actual,
                    "predicted": predicted,
                })

    with open("validation_errors.jsonl", "w", encoding="utf-8") as file:
        for error in errors:
            file.write(json.dumps(error, ensure_ascii=False) + "\n")

    counts = Counter((e["actual"], e["predicted"]) for e in errors)
    examples = defaultdict(list)
    for error in errors:
        pair = (error["actual"], error["predicted"])
        if len(examples[pair]) < 2:
            examples[pair].append(error)

    with open("failure_modes_report.md", "w", encoding="utf-8") as report:
        report.write("# Recurring Validation Failure Modes\n\n")
        report.write(f"Reviewed {len(errors)} validation errors.\n\n")

        for number, ((actual, predicted), count) in enumerate(counts.most_common(3), 1):
            report.write(
                f"## {number}. Actual `{actual}` predicted as `{predicted}` "
                f"({count} errors)\n\n"
            )
            for example in examples[(actual, predicted)]:
                report.write(
                    f"- `{example['record_id']}` — {example['name']}\n"
                    f"  - {example['description']}\n"
                )
            report.write("\n")

    print(f"Found {len(errors)} validation errors.")
    print("Saved validation_errors.jsonl and failure_modes_report.md")


if __name__ == "__main__":
    main()
