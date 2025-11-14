# JUUL Labs Collection – California

Utilities to ingest, normalize, and enrich the JUUL Labs California document collection. The dataset ships as pipe‑delimited CSVs exported from Relativity-style e‑discovery tools, so this repo focuses on taming the quirky format, aligning columns, and layering on lightweight NLP for downstream review.

## What We Built And Why

- **Schema discovery** – `inspect_headers.py` and `view_dataframe.py` let us peek at headers/rows straight from the source exports, which confirmed every file follows the same (messy) pipe layout. That validation step was essential before attempting any automation.
- **Bulk normalization** – `normalize_collection.py` walks every CSV matching `JUUL_Labs_Collection_California*.csv`, maps Relativity-style fields to a friendlier 31‑column target schema, and merges them into `merged_juul_mapped_31cols.csv`. The script also fixes common gaps (e.g., derives `FileExtension`, clones custodians, seeds empty columns) so analysts receive a consistent table regardless of the originating production.
- **Deep dive + enrichment** – `analyze_single_file.py` focuses on the largest tranche (`JUUL_Labs_Collection_California_48.csv`). In addition to the normalization above it:
  - Strips header chatter from `ocr_text` so subsequent NLP looks at message bodies.
  - Computes sentiment using NLTK’s VADER (with safe fallbacks) for reviewer triage.
  - Runs LDA topic modeling + TF‑IDF keyword extraction over `EmailContent` to surface thematic tags even when the source metadata is sparse.
  - Emits `juul_48_enriched.csv`, a ready-to-analyze table that stays git-ignored because it exceeds GitHub’s 100 MB limit; instead we store `juul_48_enriched.csv.zip`.

Together, these steps document the full pipeline from raw productions to an enriched deliverable, which is often demanded in investigations where defensible provenance is just as important as insights.

## Repository Tour

| Path | Purpose |
| --- | --- |
| `raw_data/` | Original CSV drops (kept out of version control). |
| `normalize_collection.py` | Batch normalizer that outputs `merged_juul_mapped_31cols.csv`. |
| `analyze_single_file.py` | Enriches file 48 with sentiment/topics/keywords, saving `juul_48_enriched.csv`. |
| `inspect_headers.py` | Prints each CSV’s header to confirm schema alignment. |
| `view_dataframe.py` | Quick pandas preview of any CSV (shape, columns, head). |
| `juul_48_enriched.csv.zip` | Compressed enriched dataset tracked in git. |

## Running The Tools

1. **Inspect structure (optional but recommended)**
   ```bash
   python inspect_headers.py
   python view_dataframe.py JUUL_Labs_Collection_California_01.csv
   ```
2. **Normalize entire collection**
   ```bash
   python normalize_collection.py
   # Produces merged_juul_mapped_31cols.csv
   ```
3. **Enrich tranche 48**
   ```bash
   python analyze_single_file.py
   # Produces juul_48_enriched.csv (keep local) and juul_48_enriched.csv.zip (commit-safe)
   ```

> **Note:** Scripts assume UTF‑8 input with pipe delimiters. For CSVs whose pipe rows are themselves wrapped inside a comma column, both normalization scripts fall back to a custom parser so the workflow survives inconsistent exports.

## Dependencies

- Python 3.10+
- pandas, numpy
- scikit-learn (for LDA + TF‑IDF)
- nltk (VADER sentiment; script auto-downloads the lexicon when absent)
- Optional: `vaderSentiment` package as a fallback sentiment analyzer


