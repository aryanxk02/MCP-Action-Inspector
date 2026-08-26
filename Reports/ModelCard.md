# Model Card: MCP Tool Action Classifier

## Intended use

This model classifies an MCP tool into exactly one of six action categories:

`Read`, `Write`, `Execute`, `Destructive`, `Financial`, or `Other`.

It is intended for tool inventory, routing, risk screening, and analysis. It should support human or policy review; it should not independently authorize or execute high-risk actions.

## Data and model assumptions

- Each record contains a tool `name`, natural-language `description`, and serialized `input_schema`.
- The six categories are mutually exclusive.
- Train, validation, and test records are isolated by `server_slug`.
- `server_slug` is excluded from the model input to reduce proxy leakage.
- Missing descriptions or schemas are treated as empty text.
- The model uses `distilbert-base-uncased` with a 256-token maximum input length.
- The classifier uses a fixed label mapping and deterministic `argmax` prediction.

## Limitations

- The SLM was trained for one epoch and uses no class weighting or oversampling.
- The classes are highly imbalanced: `Read` is the majority class, while `Financial` and `Other` are rare.
- Inputs longer than 256 tokens are truncated, potentially removing decisive schema fields.
- The model is not calibrated and does not provide a validated abstention or uncertainty policy.
- A predicted category describes the tool based on its text and schema; it does not verify the tool's real runtime behavior.
- Performance may degrade on new servers, unfamiliar tool styles, missing fields, or categories outside the six training labels.

## Known failure modes

Observed confusion patterns and class-level results indicate these recurring risks:

- `Execute` can be confused with `Read` or `Write` when the description does not clearly state whether an external side effect occurs.
- `Financial` can be confused with generic `Write` tools because the class is small and financial actions often look like ordinary updates.
- `Other` is difficult to identify because it is rare and heterogeneous; the validation model achieved zero F1 for this class.
- Long schemas can lose important evidence because of tail truncation.

These are evidence-based failure modes from validation metrics and confusion analysis. They should be supplemented with a documented manual review of at least 20 individual validation errors before production use.

## Evaluation and test-data warning

On the supplied validation set, the DistilBERT model achieved 0.9302 accuracy, 0.7079 macro F1, and 0.9276 weighted F1. These are validation results, not test results.

The test labels are private. Therefore, test accuracy, test F1, and true test error rates are unknown. `predictions.csv` contains predictions for the unlabeled test records only.
