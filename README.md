# MCP Tool Action Classification

This project classifies each MCP tool into one of six categories: `Read`, `Write`, `Execute`, `Destructive`, `Financial`, or `Other`.

## Setup and commands

```bash
pip install -r requirements.txt
python tfidf.py
python main.py
python analyze_validation.py
```

The training script creates the model, validation metrics, confusion matrix, and `predictions.csv`.

## Model and input representation

The SLM is `distilbert-base-uncased`, a small open-weight DistilBERT sequence-classification model. Each record is represented as:

```text
name: <tool name>
description: <tool description>
input_schema: <serialized JSON schema>
```

The `server_slug` is excluded to avoid server/category leakage. Missing fields are represented as empty strings.

## Truncation policy

Inputs are tokenized with a maximum length of 256 tokens. Longer inputs are truncated from the tail. This keeps memory usage manageable, but important fields near the end of long schemas may be lost.

## Class imbalance

The TF-IDF baseline uses balanced logistic regression with `class_weight="balanced"`. The SLM uses standard cross-entropy without class weighting or oversampling. Macro F1 is used as the primary metric so that minority classes are not hidden by the dominant `Read` class.

## Training configuration

- Epochs: 1
- Train batch size: 2 per device
- Gradient accumulation: 4 steps, effective batch size 8
- Evaluation batch size: 4
- Learning rate: `2e-5`
- Weight decay: `0.01`
- Evaluation and checkpointing: every epoch
- Checkpoint metric: validation macro F1
- Random seed: 42
- Maximum sequence length: 256

The classifier uses a fixed label mapping and `argmax`, producing exactly one deterministic valid category per record.

## Validation results

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| TF-IDF + balanced logistic regression | 0.8593 | 0.6743 | 0.8639 |
| DistilBERT SLM | 0.9302 | 0.7079 | 0.9276 |

The SLM results are measured on the supplied validation set. The test labels are private and are used only to generate `predictions.csv`.

## Limitations

- The SLM was trained for only one epoch.
- The SLM has no class-weighting or oversampling, so rare classes remain difficult.
- Tail truncation may remove important information from long schemas.
- Test accuracy cannot be measured because test labels are private.

## Inference

Run the reusable inference interface with:

```bash
python predict.py \
  --model-path slm_model/slm_model \
  --input data/test_unlabeled.jsonl \
  --output predictions.csv
```

The output contains exactly:

```csv
record_id,category
```
