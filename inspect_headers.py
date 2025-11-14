#!/usr/bin/env python3
"""
Iterate over every JUUL Labs CSV file and print its header.
Useful for verifying column names are consistent across files.
"""

from __future__ import annotations

import csv
import glob
import os
import sys
from typing import Iterable, List, Sequence

csv.field_size_limit(1_000_000_000)

DATA_GLOB = "JUUL_Labs_Collection_California*.csv"


def find_files(pattern: str) -> List[str]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No files match pattern: {pattern}")
    return paths


def read_header(path: str) -> Sequence[str]:
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="|", quotechar='"')
        try:
            header = next(reader)
        except StopIteration:
            raise SystemExit(f"{path} is empty")
    return [col.strip() for col in header]


def main() -> None:
    files = find_files(DATA_GLOB)
    baseline_header: Sequence[str] | None = None
    for path in files:
        header = read_header(path)
        if baseline_header is None:
            baseline_header = header
        else:
            if header != baseline_header:
                print(f"{path} header mismatch!", file=sys.stderr)
        print(f"{path} ({len(header)} columns)")
        for idx, name in enumerate(header, 1):
            print(f"{idx:3}: {name}")
        print("-" * 40)


if __name__ == "__main__":
    main()
