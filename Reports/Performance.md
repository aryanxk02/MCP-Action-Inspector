# Model Size and Inference Performance

## Model size

The model is `distilbert-base-uncased` adapted for six-way sequence classification.

- Main inference weights: approximately **255 MB** (`model.safetensors`).
- Saved model directory: approximately **1.0 GB** because it also contains the `checkpoint-2768` training checkpoint and related files.
- The training checkpoint and optimizer artifacts are not needed for inference.

## Inference latency and throughput

The recorded benchmark was a warm local-CPU benchmark using a 158-token sample and batch size 1:

- Single-record latency: **35.9 ms**
- Throughput: **27.9 records/second**

These figures are environment-specific. They do not represent performance on a Colab T4 GPU or production hardware. Full JSONL inference also includes file reading, tokenization, batching, and CSV writing, so its end-to-end speed can differ from this warm model-call benchmark.

## Peak memory

Peak memory was **not measured reliably** in the available run artifacts. It should be measured separately for the target deployment hardware before making a capacity or serving claim.

## Reproducibility notes

For a new benchmark, record the hardware, device, batch size, sequence length, warm-up policy, number of measured runs, and whether the timing includes tokenization and file I/O. Do not compare the values above with another environment without matching those conditions.
