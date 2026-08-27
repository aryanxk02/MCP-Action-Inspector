"""Run inference on a JSONL file and write record_id/category predictions."""

import argparse
import csv
import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABELS = ["Read", "Write", "Execute", "Destructive", "Financial", "Other"]


def load_jsonl(path):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def make_text(row):
    return "\n".join(
        [
            f"name: {row.get('name') or ''}",
            f"description: {row.get('description') or ''}",
            f"input_schema: {row.get('input_schema') or ''}",
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Predict categories for MCP tools")

    parser.add_argument(
        "--model-path",
        required=True,
    )

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    args = parser.parse_args()

    # Load tokenizer and trained model.
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)

    # Use GPU if available, otherwise CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    # Load input data.
    rows = load_jsonl(args.input)

    predictions = []

    # Run inference in batches.
    for start in range(
        0,
        len(rows),
        args.batch_size,
    ):
        batch = rows[start : start + args.batch_size]

        inputs = tokenizer(
            [make_text(row) for row in batch],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        # DistilBERT does not use token_type_ids.
        # Some tokenizers may still return it.
        inputs.pop(
            "token_type_ids",
            None,
        )

        # Move tensors to the selected device.
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Inference only; no gradients are calculated.
        with torch.inference_mode():
            logits = model(**inputs).logits

            predicted_ids = logits.argmax(dim=-1).tolist()

        predictions.extend(LABELS[index] for index in predicted_ids)

    # Write predictions to CSV.
    with open(
        args.output,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "record_id",
                "category",
            ]
        )

        writer.writerows(
            [
                [
                    row["record_id"],
                    prediction,
                ]
                for row, prediction in zip(
                    rows,
                    predictions,
                )
            ]
        )

    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
