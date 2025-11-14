#!/usr/bin/env python3
"""
Load a JUUL Labs CSV into a pandas DataFrame and preview rows/columns.
"""

from __future__ import annotations

import glob
import os
import sys
from typing import List

import csv
csv.field_size_limit(1_000_000_000)
import pandas as pd
CSV_PATTERN = "JUUL_Labs_Collection_California*.csv"


def pick_file() -> str:
    candidates: List[str] = sorted(glob.glob(CSV_PATTERN))
    if not candidates:
        raise SystemExit(f"No CSVs match pattern {CSV_PATTERN!r}")
    return candidates[0]


def load_dataframe(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        delimiter="|",
        quotechar='"',
        dtype=str,
        engine="python",
    )


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else pick_file()
    if not os.path.exists(path):
        raise SystemExit(f"File not found: {path}")
    df = load_dataframe(path)
    print(f"Loaded {path} with shape {df.shape}")
    print("\nColumns:")
    for idx, col in enumerate(df.columns, 1):
        print(f"{idx:3}: {col}")
    print("\nHead:")
    print(df.head())


if __name__ == "__main__":
    main()
