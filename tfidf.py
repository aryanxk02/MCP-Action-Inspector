"""TF-IDF + logistic regression baseline."""

import argparse
import csv
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


LABELS = ["Read", "Write", "Execute", "Destructive", "Financial", "Other"]


def load_jsonl(path):
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def make_text(record):
    # Missing fields remain empty; no information is invented.
    return " ".join([
        str(record.get("name") or ""),
        str(record.get("description") or ""),
        str(record.get("input_schema") or ""),
    ])


def calculate_metrics(y_true, y_pred):
    report = classification_report(
        y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "per_class": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
            }
            for label in LABELS
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--validation", default="data/validation.jsonl")
    parser.add_argument("--metrics", default="tfidf_metrics.json")
    parser.add_argument("--confusion-matrix", default="tfidf_confusion_matrix.csv")
    args = parser.parse_args()

    train = load_jsonl(args.train)
    validation = load_jsonl(args.validation)
    x_train = [make_text(row) for row in train]
    x_test = [make_text(row) for row in validation]
    y_train = [row["category"] for row in train]
    y_test = [row["category"] for row in validation]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = calculate_metrics(y_test, predictions)

    with open(args.metrics, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    matrix = confusion_matrix(y_test, predictions, labels=LABELS)
    with open(args.confusion_matrix, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["actual/predicted"] + LABELS)
        writer.writerows([[label] + row.tolist() for label, row in zip(LABELS, matrix)])

    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Saved: {args.metrics} and {args.confusion_matrix}")


if __name__ == "__main__":
    main()
