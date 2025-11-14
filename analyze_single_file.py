#!/usr/bin/env python3
"""
Load JUUL_Labs_Collection_California_48.csv, normalize columns, and enrich with
sentiment + topic modeling (LDA on EmailContent/ocr_text).
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

csv.field_size_limit(1_000_000_000)

import nltk
import numpy as np
import pandas as pd
from pandas.errors import ParserError
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

@dataclass(frozen=True)
class ProcessingConfig:
    csv_file: str = (
        "/Users/rutwikpatil/Desktop/Reliath/JUUL_Labs_Collection_California/raw_data/"
        "JUUL_Labs_Collection_California_48.csv"
    )
    output_file: str = "juul_48_enriched.csv"
    num_topics: int = 8
    top_words_per_topic: int = 8
    tfidf_top_n: int = 5


CONFIG = ProcessingConfig()

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
    "title": "EmailSubject",
    "datesent": "SentDate",
    "datereceived": "ReceivedDate",
    "date_modified_industry": "DateModified",
    "datesent": "DateCreated",
    "id": "FileName",
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


HEADER_PREFIXES = (
    "from:",
    "to:",
    "cc:",
    "bcc:",
    "subject:",
    "sent:",
    "received:",
    "date:",
    "time:",
    "attachments:",
    "attachment:",
    "conversation:",
    "view in browser",
)

HEADER_ONLY_LINE = re.compile(r"^([A-Za-z ]+: ?)+$", re.IGNORECASE)


def clean_email_body(value: str | float) -> str | float:
    if not isinstance(value, str):
        return value
    text = value.replace("\r", "")
    lines = text.split("\n")
    body_lines: List[str] = []
    removing_header = True
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if removing_header:
            if not stripped:
                continue
            if HEADER_ONLY_LINE.match(stripped):
                continue
            if any(lower.startswith(prefix) for prefix in HEADER_PREFIXES):
                continue
            removing_header = False
        body_lines.append(stripped)
    cleaned = "\n".join(body_lines).strip()
    return cleaned if cleaned else value.strip()


def _read_wrapped_pipe_rows(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as handle:
        comma_reader = csv.reader(handle, delimiter=",", quotechar='"')
        for row in comma_reader:
            if not row:
                continue
            yield row[0]


def _parse_pipe_row(raw: str) -> List[str]:
    return next(csv.reader([raw], delimiter="|", quotechar='"'))


def _load_wrapped_pipe_dataframe(path: str, usecols: List[str]) -> pd.DataFrame:
    row_iter = _read_wrapped_pipe_rows(path)
    try:
        header_raw = next(row_iter)
    except StopIteration:
        return pd.DataFrame(columns=usecols)
    header = _parse_pipe_row(header_raw)
    index = {name: idx for idx, name in enumerate(header)}
    data: Dict[str, List[str]] = {col: [] for col in usecols}
    for raw_row in row_iter:
        row = _parse_pipe_row(raw_row)
        for col in usecols:
            idx = index.get(col)
            value = row[idx] if idx is not None and idx < len(row) else None
            data[col].append(value)
    return pd.DataFrame(data, columns=usecols)


def load_dataframe(path: str, usecols: List[str]) -> pd.DataFrame:
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
        print(f"ParserError for {path}: {exc}. Attempting wrapped-pipe fallback.")
        return _load_wrapped_pipe_dataframe(path, usecols)


def transform_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COLUMN_MAP)
    text_series = df.get("TextPath", pd.Series(index=df.index, dtype="object"))
    df["EmailContent"] = text_series.map(clean_email_body)
    if "Document Types" in df.columns:
        df["FileType"] = df["Document Types"]
    if "FileName" in df.columns:
        splits = df["FileName"].astype(str).str.rsplit(".", n=1)
        df["FileExtension"] = splits.str[-1].where(splits.str.len() > 1).str.lower()
    if "Custodian" in df.columns:
        df["Principal Custodian"] = df["Custodian"]
    for col in TARGET_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[TARGET_COLS]


def _get_sentiment_analyzer():
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer as NltkSIA  # type: ignore
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            try:
                nltk.download("vader_lexicon", quiet=True)
            except Exception:
                pass
        try:
            return NltkSIA()
        except LookupError:
            pass
    except ImportError:
        pass

    try:
        from vaderSentiment.vaderSentiment import (  # type: ignore
            SentimentIntensityAnalyzer as VaderSIA,
        )

        return VaderSIA()
    except ImportError:
        pass

    class SimpleSentiment:
        POSITIVE = {
            "good",
            "great",
            "excellent",
            "positive",
            "success",
            "win",
            "improve",
            "benefit",
            "happy",
            "joy",
            "love",
        }
        NEGATIVE = {
            "bad",
            "terrible",
            "awful",
            "negative",
            "fail",
            "risk",
            "issue",
            "problem",
            "concern",
            "angry",
            "sad",
        }

        def polarity_scores(self, text: str) -> Dict[str, float]:
            text = text.lower()
            words = text.split()
            pos = sum(1 for w in words if w in self.POSITIVE)
            neg = sum(1 for w in words if w in self.NEGATIVE)
            total = pos + neg
            compound = 0.0 if total == 0 else (pos - neg) / total
            return {"compound": compound}

    print("Warning: falling back to simple lexicon-based sentiment analyzer.")
    return SimpleSentiment()


def compute_sentiment(text_series: pd.Series) -> pd.Series:
    sia = _get_sentiment_analyzer()

    def label(text: str | float) -> str:
        if not isinstance(text, str) or not text.strip():
            return "neutral (0.00)"
        score = sia.polarity_scores(text)["compound"]
        if score >= 0.05:
            mood = "positive"
        elif score <= -0.05:
            mood = "negative"
        else:
            mood = "neutral"
        return f"{mood} ({score:.2f})"

    return text_series.fillna("").map(label)


def run_lda(text_series: pd.Series) -> Tuple[pd.Series, Dict[int, str]]:
    texts = text_series.fillna("").astype(str)
    mask = texts.str.strip().astype(bool)
    if not mask.any():
        return pd.Series(pd.NA, index=text_series.index), {}
    vectorizer = CountVectorizer(
        stop_words="english",
        max_df=0.8,
        min_df=5,
        max_features=5000,
    )
    dtm = vectorizer.fit_transform(texts[mask])
    if dtm.shape[0] == 0:
        return pd.Series(pd.NA, index=text_series.index), {}
    n_topics = min(CONFIG.num_topics, dtm.shape[0])
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=10,
        learning_method="online",
        random_state=42,
    )
    topic_matrix = lda.fit_transform(dtm)
    feature_names = vectorizer.get_feature_names_out()

    topic_labels: Dict[int, str] = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[:-CONFIG.top_words_per_topic - 1:-1]
        words = [feature_names[i] for i in top_indices]
        topic_labels[topic_idx] = ", ".join(words)

    doc_topics = []
    for row in topic_matrix:
        topic_idx = int(np.argmax(row))
        doc_topics.append(topic_labels.get(topic_idx, pd.NA))

    result = pd.Series(pd.NA, index=text_series.index, dtype="object")
    result.loc[mask] = doc_topics
    return result, topic_labels


def extract_keywords_tfidf(text_series: pd.Series, top_n: Optional[int] = None) -> pd.Series:
    if top_n is None:
        top_n = CONFIG.tfidf_top_n
    texts = text_series.fillna("").astype(str)
    if not texts.str.strip().any():
        return pd.Series(pd.NA, index=text_series.index)
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    keywords: List[Optional[str]] = []
    for row_idx in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix.getrow(row_idx)
        if row.nnz == 0:
            keywords.append(pd.NA)
            continue
        pairs = sorted(
            zip(row.indices, row.data),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]
        terms = [feature_names[idx] for idx, _ in pairs]
        keywords.append(", ".join(terms))
    return pd.Series(keywords, index=text_series.index, dtype="object")


def main() -> None:
    if not os.path.exists(CONFIG.csv_file):
        raise SystemExit(f"Missing source file: {CONFIG.csv_file}")
    usecols = list(COLUMN_MAP.keys())
    raw_df = load_dataframe(CONFIG.csv_file, usecols)
    normalized = transform_dataframe(raw_df)
    normalized["Sentiment"] = compute_sentiment(normalized["EmailContent"])
    lda_topics, topic_dict = run_lda(normalized["EmailContent"])
    normalized["Topics"] = lda_topics
    tfidf_topics = extract_keywords_tfidf(normalized["EmailContent"])
    normalized["Key/Main Topics"] = tfidf_topics
    missing_topics = normalized["Key/Main Topics"].isna()
    if missing_topics.any():
        normalized.loc[missing_topics, "Key/Main Topics"] = normalized.loc[missing_topics, "Topics"]
    normalized.to_csv(CONFIG.output_file, index=False)
    print(f"Saved enriched dataset to {CONFIG.output_file} ({len(normalized)} rows)")
    if topic_dict:
        print("\nTop words per topic:")
        for idx, words in topic_dict.items():
            print(f"Topic {idx}: {words}")


if __name__ == "__main__":
    main()
