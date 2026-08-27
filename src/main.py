"""Fine-tune DistilBERT with minority-class oversampling and weighted loss."""

import csv
import json
import random

import numpy as np
import torch
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

# ---------------------------------------------------------
# OVERSAMPLING SETTINGS
# ---------------------------------------------------------
#
# These are the TARGET numbers for the training set.
#
# Financial originally has 111 examples.
# Other originally has 76 examples.
#
# We will repeat existing examples until we reach these
# target counts. No synthetic/fake data is generated.
#
OVERSAMPLE_TARGETS = {
    "Financial": 500,
    "Other": 300,
}


def load_data(path):
    """Load JSONL data into a list of dictionaries."""
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def make_text(row):
    """
    Convert a tool record into the text given to DistilBERT.
    """
    return "\n".join(
        [
            f"name: {row.get('name') or ''}",
            f"description: {row.get('description') or ''}",
            f"input_schema: {row.get('input_schema') or ''}",
        ]
    )


# ---------------------------------------------------------
# OVERSAMPLING
# ---------------------------------------------------------


def oversample_minority_classes(rows, target_counts, seed=42):
    """
    Oversample selected classes by repeating existing examples.

    IMPORTANT:
    This does NOT create fake/synthetic data.

    Example:
        Financial = [A, B, C]

    Target = 6

    Result could be:
        [A, B, C, A, C, B]

    Every example is still an original training example.
    """

    rng = random.Random(seed)

    # Group examples by category.
    grouped = {}

    for row in rows:
        category = row["category"]
        grouped.setdefault(category, []).append(row)

    oversampled_rows = []

    for category, examples in grouped.items():
        # If a category isn't specified in target_counts,
        # keep its original size.
        target = target_counts.get(
            category,
            len(examples),
        )

        # Never remove examples.
        if target <= len(examples):
            selected_examples = examples

        else:
            # Keep every original example once.
            selected_examples = list(examples)

            # Then randomly repeat existing examples
            # until the desired target is reached.
            additional_examples = rng.choices(
                examples,
                k=target - len(examples),
            )

            selected_examples.extend(additional_examples)

        oversampled_rows.extend(selected_examples)

    # Shuffle so repeated minority examples aren't grouped together.
    rng.shuffle(oversampled_rows)

    return oversampled_rows


def calculate_class_weights(rows):
    """
    Calculate balanced class weights from the ACTUAL
    training distribution after oversampling.

    Formula:

        weight = N / (K * class_count)

    where:

        N = total number of training examples
        K = number of classes
        class_count = number of examples in that class

    We normalize the weights so their average is 1.0.
    """

    class_counts = {label: 0 for label in LABELS}

    for row in rows:
        category = row["category"]

        if category not in class_counts:
            raise ValueError(f"Unknown category found: {category}")

        class_counts[category] += 1

    total_samples = len(rows)
    num_classes = len(LABELS)

    # Standard balanced weighting.
    weights = {
        label: total_samples / (num_classes * class_counts[label]) for label in LABELS
    }

    # Normalize so the average weight is 1.
    mean_weight = sum(weights.values()) / num_classes

    weights = {label: weight / mean_weight for label, weight in weights.items()}

    return class_counts, weights


# ---------------------------------------------------------
# DATASET
# ---------------------------------------------------------


class ToolDataset(Dataset):
    def __init__(
        self,
        rows,
        tokenizer,
        labelled=True,
    ):
        # Tokenize the entire text.
        self.data = tokenizer(
            [make_text(row) for row in rows],
            truncation=True,
            max_length=MAX_LENGTH,
        )

        # Encode labels:
        #
        # Read        -> 0
        # Write       -> 1
        # Execute     -> 2
        # Destructive -> 3
        # Financial   -> 4
        # Other       -> 5
        #
        if labelled:
            self.data["labels"] = [LABEL_TO_ID[row["category"]] for row in rows]

    def __len__(self):
        return len(self.data["input_ids"])

    def __getitem__(self, index):
        return {key: value[index] for key, value in self.data.items()}


# ---------------------------------------------------------
# WEIGHTED LOSS TRAINER
# ---------------------------------------------------------


class WeightedLossTrainer(Trainer):
    """
    Hugging Face Trainer using class-weighted
    CrossEntropyLoss.

    This makes mistakes on minority classes more costly.
    """

    def __init__(
        self,
        *args,
        class_weights=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if class_weights is None:
            class_weights = [1.0 for _ in LABELS]

        self.class_weights = torch.tensor(
            class_weights,
            dtype=torch.float,
        )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        # Remove labels before passing inputs to model.
        labels = inputs.pop("labels")

        outputs = model(**inputs)

        logits = outputs.logits

        # Put class weights on the same device as logits.
        weights = self.class_weights.to(logits.device)

        loss_function = torch.nn.CrossEntropyLoss(weight=weights)

        loss = loss_function(
            logits,
            labels,
        )

        if return_outputs:
            return loss, outputs

        return loss


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------


def compute_metrics(result):

    # Select the class with the highest logit.
    predictions = np.argmax(
        result.predictions,
        axis=-1,
    )

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


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------


def main():

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    train_rows = load_data("data/train.jsonl")

    validation_rows = load_data("data/validation.jsonl")

    test_rows = load_data("data/test_unlabeled.jsonl")

    print(f"Original training examples: {len(train_rows)}")

    print(f"Validation examples: {len(validation_rows)}")

    print(f"Test examples: {len(test_rows)}")

    # -----------------------------------------------------
    # OVERSAMPLE TRAINING DATA
    # -----------------------------------------------------
    #
    # IMPORTANT:
    # Only train_rows are oversampled.
    #
    # validation_rows and test_rows remain untouched.
    #

    train_rows = oversample_minority_classes(
        train_rows,
        OVERSAMPLE_TARGETS,
        seed=42,
    )

    print(f"Oversampled training examples: {len(train_rows)}")

    # -----------------------------------------------------
    # CALCULATE CLASS WEIGHTS
    # -----------------------------------------------------
    #
    # Calculate weights AFTER oversampling.
    #

    class_counts, class_weights = calculate_class_weights(train_rows)

    print("\nTraining distribution after oversampling:")

    for label in LABELS:
        print(f"  {label:12s}: {class_counts[label]}")

    print("\nCalculated class weights:")

    for label in LABELS:
        print(f"  {label:12s}: {class_weights[label]:.4f}")

    # Convert dictionary into the exact label order
    # expected by CrossEntropyLoss.
    class_weights_list = [class_weights[label] for label in LABELS]

    # -----------------------------------------------------
    # TOKENIZER
    # -----------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id=LABEL_TO_ID,
    )

    # -----------------------------------------------------
    # DATASETS
    # -----------------------------------------------------

    train_data = ToolDataset(
        train_rows,
        tokenizer,
    )

    validation_data = ToolDataset(
        validation_rows,
        tokenizer,
    )

    test_data = ToolDataset(
        test_rows,
        tokenizer,
        labelled=False,
    )

    # -----------------------------------------------------
    # TRAINER
    # -----------------------------------------------------

    trainer = WeightedLossTrainer(
        model=model,
        class_weights=class_weights_list,
        args=TrainingArguments(
            output_dir="slm_model",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
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

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    print("\nStarting training...")

    trainer.train()

    # -----------------------------------------------------
    # SAVE MODEL
    # -----------------------------------------------------

    trainer.save_model("slm_model")

    tokenizer.save_pretrained("slm_model")

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    print("\nEvaluating validation set...")

    result = trainer.predict(validation_data)

    predictions = np.argmax(
        result.predictions,
        axis=-1,
    )

    report = classification_report(
        result.label_ids,
        predictions,
        labels=range(len(LABELS)),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )

    # Add confusion matrix.
    report["confusion_matrix"] = confusion_matrix(
        result.label_ids,
        predictions,
        labels=range(len(LABELS)),
    ).tolist()

    # Save metrics.
    with open(
        "slm_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    # -----------------------------------------------------
    # TEST PREDICTIONS
    # -----------------------------------------------------

    print("\nGenerating test predictions...")

    test_predictions = np.argmax(
        trainer.predict(test_data).predictions,
        axis=-1,
    )

    with open(
        "predictions.csv",
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
                    LABELS[prediction],
                ]
                for row, prediction in zip(
                    test_rows,
                    test_predictions,
                )
            ]
        )

    # -----------------------------------------------------
    # FINAL RESULTS
    # -----------------------------------------------------

    print("\nValidation results:")

    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "accuracy",
                    "macro avg",
                    "weighted avg",
                )
            },
            indent=2,
        )
    )

    print("\nPer-class F1:")

    for label in LABELS:
        print(f"  {label:12s}: {report[label]['f1-score']:.4f}")


if __name__ == "__main__":
    main()
