# Data Audit

The figures below come from `eda.py`, run on the supplied JSONL files. The EDA script treats a field as missing when `record.get(field)` is falsey. It checks JSONL parsing, parses non-empty `input_schema` values as JSON objects, counts exact duplicate feature rows, and reports text lengths.

## Label counts and imbalance

| Split | Read | Write | Execute | Destructive | Financial | Other | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 14,084 (63.60%) | 5,433 (24.54%) | 1,175 (5.31%) | 1,264 (5.71%) | 111 (0.50%) | 76 (0.34%) | 22,143 |
| Validation | 2,817 (63.60%) | 1,087 (24.54%) | 235 (5.31%) | 253 (5.71%) | 22 (0.50%) | 15 (0.34%) | 4,429 |

The data is strongly imbalanced. `Read` is the majority class, while `Financial` and `Other` are rare. The test set is unlabeled, so no test label counts are available.

## Missing and malformed fields

| Split | Missing descriptions | Missing schemas | Malformed JSONL lines | Malformed non-empty schemas |
|---|---:|---:|---:|---:|
| Train | 316 (1.43%) | 3,617 (16.33%) | 0 | 0 |
| Validation | 6 (0.14%) | 737 (16.64%) | 0 | 0 |
| Test | 42 (0.95%) | 743 (16.78%) | 0 | 0 |

`record_id`, `server_slug`, and `name` are present in every record in all three splits according to `eda.py`. A missing schema is not counted as malformed; it is represented as an empty value by the modeling code. Every non-empty schema checked by `eda.py` parsed as a JSON object.

## Duplicate risks

`eda.py` counts exact duplicate feature rows using the tuple `(name, description, input_schema)`. The reported counts include all rows belonging to duplicate groups:

| Split | Rows in exact duplicate feature groups |
|---|---:|
| Train | 1,108 |
| Validation | 62 |
| Test | 183 |

These duplicates are a risk because repeated tool text can make evaluation less representative. This audit does not determine whether duplicate rows have conflicting labels, and it does not perform near-duplicate or semantic-similarity detection.

## Leakage risks

The supplied split is grouped by `server_slug`. A separate check found zero server overlap between train and validation, train and test, or validation and test. This reduces leakage from related tools on the same server. No new random row-level split was created.

`server_slug` is excluded from the model input because it could act as a proxy for the category. The model uses only `name`, `description`, and `input_schema`. This audit does not prove that the text contains no label-like wording; it only identifies the main structural leakage risk and keeps the server key out of the features. The held-out test labels were not used for tuning.

## Why these input fields were used

- `name` often contains a concise operation or action, such as `get`, `update`, `delete`, or `execute`.
- `description` provides the tool's intended behavior and can reveal whether it reads data, changes state, executes an operation, or performs a financial action.
- `input_schema` provides structured argument names and descriptions that may distinguish read operations from writes, destructive operations, or executions.

Missing values are kept as empty strings rather than replaced with invented information. The three fields are combined into labeled text before tokenization. The input excludes `record_id` and `server_slug` because they identify or group records rather than describe the tool's action.

## Text-length observation

The combined character lengths reported by `eda.py` were:

| Split | Minimum | Median | Mean | Maximum |
|---|---:|---:|---:|---:|
| Train | 6 | 372 | 622.9 | 40,907 |
| Validation | 8 | 408 | 693.0 | 31,064 |
| Test | 6 | 339 | 547.6 | 15,269 |

The model uses a 256-token limit, so long records may be truncated. This is a known information-loss risk, especially for schemas whose decisive fields occur near the end.
