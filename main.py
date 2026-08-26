"""Fine-tune DistilBERT and create validation metrics and test predictions."""

import csv
import json

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


LABELS = ["Read", "Write", "Execute", "Destructive", "Financial", "Other"]
LABEL_TO_ID = {label: number for number, label in enumerate(LABELS)}
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256


def load_data(path):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()] # returns a list of dictionaries line by line

"""
Input
row = {
    "name": "X",
    "description": "Get information.",
    "input_schema": "{\"type\":\"object\"}"
}
Output
  name: X
  description: Get information.
  input_schema: {"type":"object"}
"""
def make_text(row):
    return "\n".join([
        f"name: {row.get('name') or ''}",
        f"description: {row.get('description') or ''}",
        f"input_schema: {row.get('input_schema') or ''}",
    ])


class ToolDataset(Dataset):
    def __init__(self, rows, tokenizer, labelled=True):
        # tokenizes the entire text (line by line)
        self.data = tokenizer(
            [make_text(row) for row in rows],
            truncation=True,
            max_length=MAX_LENGTH, # 256
        )
        # encodes the targets (Read, Write, etc) to 0, 1, 2...
        if labelled:
            self.data["labels"] = [LABEL_TO_ID[row["category"]] for row in rows]

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, index):
        return {key: value[index] for key, value in self.data.items()}


def compute_metrics(result):
    # returns the prediction with the highest value
    predictions = np.argmax(result.predictions, axis=-1)
    report = classification_report(
        result.label_ids,
        predictions,
        labels=range(len(LABELS)),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }


def main():
    train_rows = load_data("data/train.jsonl")
    validation_rows = load_data("data/validation.jsonl")
    test_rows = load_data("data/test_unlabeled.jsonl")

    # tokenizer converts text into token IDs, which is understood by the model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # slm model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS), # 6
        id2label=dict(enumerate(LABELS)),
        label2id=LABEL_TO_ID, # reverse mapping
    )

    train_data = ToolDataset(train_rows, tokenizer)
    validation_data = ToolDataset(validation_rows, tokenizer)
    test_data = ToolDataset(test_rows, tokenizer, labelled=False)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="slm_model",
            num_train_epochs=1,
            per_device_train_batch_size=2, # processes 2 training examples at a time
            per_device_eval_batch_size=4, # processes 4 validation examples at a time
            gradient_accumulation_steps=4, # updates model weights after 4 batches
            learning_rate=2e-5,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            save_total_limit=1,
            logging_steps=100,
            report_to="none",
            seed=42,
            fp16=False,
        ),
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("slm_model")
    tokenizer.save_pretrained("slm_model")

    result = trainer.predict(validation_data)
    predictions = np.argmax(result.predictions, axis=-1)
    report = classification_report(
        result.label_ids,
        predictions,
        labels=range(len(LABELS)),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    report["confusion_matrix"] = confusion_matrix(
        result.label_ids, predictions, labels=range(len(LABELS))
    ).tolist()
    with open("slm_metrics.json", "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    test_predictions = np.argmax(trainer.predict(test_data).predictions, axis=-1)
    with open("predictions.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["record_id", "category"])
        writer.writerows([
            [row["record_id"], LABELS[prediction]]
            for row, prediction in zip(test_rows, test_predictions)
        ])

    print(json.dumps({key: report[key] for key in
                      ("accuracy", "macro avg", "weighted avg")}, indent=2))


if __name__ == "__main__":
    main()
