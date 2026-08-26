"""Basic exploratory data analysis for the MCP classification data."""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


LABELS = ["Read", "Write", "Execute", "Destructive", "Financial", "Other"]
FIELDS = ["record_id", "server_slug", "name", "description", "input_schema"]


def load_jsonl(path):
    records = []
    bad_lines = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            bad_lines.append(number)
    return records, bad_lines


def text_length(record):
    values = [record.get(field) or "" for field in ("name", "description", "input_schema")]
    return len(" ".join(map(str, values)))


def report(path):
    records, bad_lines = load_jsonl(path)
    print(f"\n{path}: {len(records)} rows")
    print(f"Malformed JSONL lines: {len(bad_lines)}")

    for field in FIELDS:
        missing = sum(not record.get(field) for record in records)
        print(f"Missing {field}: {missing} ({missing / len(records):.2%})")

    if any("category" in record for record in records):
        counts = Counter(record.get("category") for record in records)
        print("Labels:", {label: counts.get(label, 0) for label in LABELS})

    schemas = []
    for record in records:
        schema = record.get("input_schema")
        if schema:
            try:
                parsed = json.loads(schema)
                if not isinstance(parsed, dict):
                    schemas.append(False)
                else:
                    schemas.append(True)
            except (TypeError, json.JSONDecodeError):
                schemas.append(False)
    print("Malformed input_schema values:", schemas.count(False))

    keys = [
        (record.get("name"), record.get("description"), record.get("input_schema"))
        for record in records
    ]
    duplicate_rows = sum(count for count in Counter(keys).values() if count > 1)
    print("Duplicate feature rows:", duplicate_rows)

    lengths = [text_length(record) for record in records]
    print(
        "Text length (characters):"
        f" min={min(lengths)}, median={statistics.median(lengths)},"
        f" mean={statistics.mean(lengths):.1f}, max={max(lengths)}"
    )
    print("Unique servers:", len({record.get("server_slug") for record in records}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", default=[
        "data/train.jsonl", "data/validation.jsonl", "data/test_unlabeled.jsonl"
    ])
    args = parser.parse_args()
    for path in args.files:
        report(path)


if __name__ == "__main__":
    main()
