#!/usr/bin/env python3
"""
Normalize all JUUL Labs CSVs into a unified column layout and export a merged file.
"""

from __future__ import annotations

import glob
import os
import sys
from typing import Dict, Iterable, List

import csv

csv.field_size_limit(1_000_000_000)
import pandas as pd
from pandas.errors import ParserError

CSV_PATTERN = "JUUL_Labs_Collection_California*.csv"

COLUMN_MAP = {
    "case": "Related Case",
    "bates": "BegDoc",
    "bates_alternate": "EndDoc",
    "attachmentnum": "BegAtt",
    "attachment": "EndAtt",
    "custodian": "Custodian",
    "author": "FROM",
    "recipient": "TO",
    "copied": "CC",
    "mentioned": "BCC",
    "subject": "EmailSubject",
    "datesent": "SentDate",
    "datereceived": "ReceivedDate",
    "date_modified_industry": "DateModified",
    "date_added_industry": "DateCreated",
    "filename": "FileName",
    "format": "FileType",
    "filepath": "Location",
    "ocr_text": "TextPath",
    "language": "Language",
    "keywords": "Key/Main Topics",
    "topic": "Topics",
    "type": "Document Types",
}

TARGET_COLS: List[str] = [
    "Related Case",
    "File itself for downloading",
    "BegDoc",
    "EndDoc",
    "BegAtt",
    "EndAtt",
    "Custodian",
    "FROM",
    "TO",
    "CC",
    "BCC",
    "EmailSubject",
    "SentDate",
    "ReceivedDate",
    "DateModified",
    "DateCreated",
    "FileName",
    "FileType",
    "FileExtension",
    "MD5Hash",
    "Location",
    "Principal Custodian",
    "Document Types",
    "Language",
    "Sentiment",
    "Topics",
    "Key/Main Topics",
    "EmailContent",
]


def pick_files(pattern: str = CSV_PATTERN) -> List[str]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No CSVs match pattern {pattern!r}")
    print(f"Found {len(files)} files.")
    return files


def _read_wrapped_pipe_rows(path: str) -> Iterable[str]:
    """Yield pipe-delimited rows extracted from a comma-delimited file."""
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as handle:
        comma_reader = csv.reader(handle, delimiter=",", quotechar='"')
        for row in comma_reader:
            if not row:
                continue
            yield row[0]


def _parse_pipe_row(raw: str) -> List[str]:
    """Parse a single pipe-delimited row represented as a string."""
    return next(csv.reader([raw], delimiter="|", quotechar='"'))


def _load_wrapped_pipe_dataframe(path: str, usecols: List[str]) -> pd.DataFrame:
    """Handle files where the entire pipe row is stored in the first comma column."""
    row_iter = _read_wrapped_pipe_rows(path)
    try:
        raw_header = next(row_iter)
    except StopIteration:
        return pd.DataFrame(columns=usecols)
    header = _parse_pipe_row(raw_header)
    header_index: Dict[str, int] = {name: idx for idx, name in enumerate(header)}
    data: Dict[str, List[str]] = {col: [] for col in usecols}
    for raw_row in row_iter:
        row = _parse_pipe_row(raw_row)
        for col in usecols:
            idx = header_index.get(col)
            value = row[idx] if idx is not None and idx < len(row) else None
            data[col].append(value)
    return pd.DataFrame(data, columns=usecols)


def load_dataframe(path: str, usecols: List[str] | None = None) -> pd.DataFrame:
    try:
        return pd.read_csv(
            path,
            delimiter="|",
            quotechar='"',
            dtype=str,
            engine="python",
            usecols=usecols,
        )
    except ParserError as exc:
        if usecols is None:
            raise
        print(f"ParserError for {path}: {exc}. Attempting wrapped-pipe fallback.")
        return _load_wrapped_pipe_dataframe(path, usecols)


def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COLUMN_MAP)
    if "FileName" in df.columns:
        exts = df["FileName"].astype(str).str.rsplit(".", n=1)
        df["FileExtension"] = exts.str[-1].where(exts.str.len() > 1).str.lower()
    if "Custodian" in df.columns and "Principal Custodian" not in df.columns:
        df["Principal Custodian"] = df["Custodian"]
    if "TextPath" in df.columns and "EmailContent" not in df.columns:
        df["EmailContent"] = df["TextPath"]
    for col in TARGET_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[TARGET_COLS]


def process_files(files: List[str], output_path: str) -> None:
    usecols = list(COLUMN_MAP.keys())
    first = True
    total_rows = 0
    for path in files:
        print(f"Processing {path} ...")
        raw_df = load_dataframe(path, usecols=usecols)
        df = transform_dataframe(raw_df)
        total_rows += len(df)
        df.to_csv(
            output_path,
            mode="w" if first else "a",
            header=first,
            index=False,
        )
        first = False
        print(f"  -> wrote {len(df)} rows (running total {total_rows})")
    print(f"\nDone. Total rows written: {total_rows}")
    print(f"Output saved to {output_path}")


def main() -> None:
    pattern_or_file = sys.argv[1] if len(sys.argv) > 1 else CSV_PATTERN
    files = [pattern_or_file] if os.path.isfile(pattern_or_file) else pick_files(pattern_or_file)
    output = "merged_juul_mapped_31cols.csv"
    if os.path.exists(output):
        os.remove(output)
    process_files(files, output)


if __name__ == "__main__":
    main()
