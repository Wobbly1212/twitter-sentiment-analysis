"""Convert labeled TweetClaw exports into Sentiment140-compatible CSV rows."""

import argparse
import csv
import json
from pathlib import Path


TEXT_FIELDS = ("text", "content", "full_text", "tweet_text", "body")
ID_FIELDS = ("tweet_id", "tweetId", "id_str", "id")
DATE_FIELDS = ("created_at", "createdAt", "date", "timestamp")
USER_FIELDS = ("username", "user", "handle", "author", "screen_name")
LABEL_FIELDS = ("sentiment", "label", "target", "polarity")
POSITIVE_LABELS = {"4", "1", "positive", "pos"}
NEGATIVE_LABELS = {"0", "negative", "neg"}


def first_value(row, fields, default=""):
    for field in fields:
        value = row.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def normalize_target(value):
    label = str(value).strip().lower()
    if label in POSITIVE_LABELS:
        return "4"
    if label in NEGATIVE_LABELS:
        return "0"
    return None


def read_json_records(path):
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        return data if isinstance(data, list) else [data]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def read_records(path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    if suffix in {".json", ".jsonl", ".ndjson"}:
        return read_json_records(path)
    raise ValueError(f"Unsupported input extension: {path.suffix}")


def convert_records(records, default_query="NO_QUERY"):
    converted = []
    skipped = 0
    for index, row in enumerate(records, start=1):
        target = normalize_target(first_value(row, LABEL_FIELDS))
        text = first_value(row, TEXT_FIELDS)
        if target is None or not text:
            skipped += 1
            continue
        converted.append(
            [
                target,
                first_value(row, ID_FIELDS, default=f"tweetclaw-row-{index}"),
                first_value(row, DATE_FIELDS),
                default_query,
                first_value(row, USER_FIELDS, default="unknown"),
                text,
            ]
        )
    return converted, skipped


def write_sentiment140(rows, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert labeled TweetClaw JSON, JSONL, NDJSON, or CSV exports to Sentiment140 CSV."
    )
    parser.add_argument("input", type=Path, help="Reviewed TweetClaw export with human sentiment labels")
    parser.add_argument("output", type=Path, help="Destination Sentiment140-style CSV")
    parser.add_argument("--query", default="NO_QUERY", help="Value for the Sentiment140 query column")
    return parser.parse_args()


def main():
    args = parse_args()
    rows, skipped = convert_records(read_records(args.input), default_query=args.query)
    write_sentiment140(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}; skipped {skipped} unlabeled or empty rows.")


if __name__ == "__main__":
    main()
