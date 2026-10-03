"""Create traceable sentence and clause tables for X and Guardian news."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import spacy
from spacy.language import Language
from spacy.tokens import Span


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_X_CSV = PROJECT_ROOT / "data" / "x_ai_news_30d" / "normalized" / "news_stories.csv"
DEFAULT_GUARDIAN_CSV = PROJECT_ROOT / "data" / "guardian_ai_2026_09" / "normalized" / "guardian_articles.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_news"
MODEL_NAME = "en_core_web_sm"
SPLIT_RULE_VERSION = "dependency_boundaries_v2"


class ArticleTextExtractor(HTMLParser):
    """Extract readable text while retaining paragraph-like boundaries."""

    BLOCK_TAGS = {
        "address", "article", "aside", "blockquote", "br", "div", "figcaption",
        "footer", "h1", "h2", "h3", "h4", "h5", "h6", "header", "li", "main",
        "p", "section", "table", "td", "th", "tr",
    }
    SKIP_TAGS = {"script", "style", "noscript", "svg", "gu-atom"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
        elif self.skip_depth == 0 and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
        elif self.skip_depth == 0 and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth == 0:
            self.parts.append(data)


def extract_html_text(value: str) -> str:
    parser = ArticleTextExtractor()
    parser.feed(value)
    parser.close()
    raw = html.unescape("".join(parser.parts)).replace("\xa0", " ")
    lines = [re.sub(r"\s+", " ", line).strip() for line in raw.splitlines()]
    return "\n\n".join(line for line in lines if line)


def clean_plain_text(value: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in value.splitlines()]
    return "\n\n".join(line for line in lines if line)


def combine_title_and_content(title: str, content: str) -> tuple[str, int]:
    clean_title = clean_plain_text(title)
    clean_content = clean_plain_text(content)
    if clean_title and clean_content:
        return f"{clean_title}\n\n{clean_content}", len(clean_title)
    if clean_title:
        return clean_title, len(clean_title)
    return clean_content, 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_documents(x_csv: Path, guardian_csv: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for row in read_csv(x_csv):
        news_id = row["news_id"].strip()
        text, title_end = combine_title_and_content(row.get("headline", ""), row.get("summary", ""))
        if not news_id or not text:
            continue
        documents.append(
            {
                "document_id": f"x::{news_id}",
                "source_id": news_id,
                "platform": "x",
                "title": clean_plain_text(row.get("headline", "")),
                "published_at": row.get("event_time") or row.get("updated_at", ""),
                "time_basis": row.get("event_time_basis", "x_news_updated_at"),
                "content_kind": "headline_summary",
                "matched_queries": row.get("matched_queries", ""),
                "source_path": x_csv.relative_to(PROJECT_ROOT).as_posix(),
                "document_text": text,
                "title_end_char": title_end,
                "content_segment_name": "summary",
            }
        )

    for row in read_csv(guardian_csv):
        guardian_id = row["guardian_id"].strip()
        body_path_value = row.get("body_html_path", "").strip()
        body_path = Path(body_path_value)
        if not body_path.is_absolute():
            body_path = PROJECT_ROOT / body_path
        if not guardian_id or not body_path.exists():
            continue
        body = extract_html_text(body_path.read_text(encoding="utf-8"))
        text, title_end = combine_title_and_content(row.get("web_title", ""), body)
        if not text:
            continue
        documents.append(
            {
                "document_id": f"guardian::{guardian_id}",
                "source_id": guardian_id,
                "platform": "guardian",
                "title": clean_plain_text(row.get("web_title", "")),
                "published_at": row.get("web_publication_date", ""),
                "time_basis": "guardian_web_publication_date",
                "content_kind": "title_full_body",
                "matched_queries": row.get("matched_queries", ""),
                "source_path": body_path.relative_to(PROJECT_ROOT).as_posix(),
                "document_text": text,
                "title_end_char": title_end,
                "content_segment_name": "body",
            }
        )
    return documents


def source_segment(start: int, end: int, title_end: int, content_name: str) -> str:
    if title_end and end <= title_end:
        return "title"
    if title_end and start < title_end < end:
        return "mixed"
    return content_name


def alpha_word_count(tokens: Sequence[Any]) -> int:
    return sum(1 for token in tokens if token.is_alpha)


def candidate_clause_starts(sentence: Span) -> dict[int, str]:
    starts: dict[int, str] = {}
    for token in sentence:
        if token.i == sentence.start:
            continue
        lower = token.lower_
        if token.dep_ == "cc" and lower in {"and", "but", "or", "yet", "so", "nor"}:
            head = token.head
            if head.pos_ in {"VERB", "AUX"} or head.dep_ == "conj":
                starts[token.i] = "coordinating_conjunction"
        elif token.dep_ == "mark" and token.head.dep_ in {"advcl", "ccomp"}:
            starts[token.i] = "subordinate_marker"
        elif token.dep_ in {"advcl", "parataxis"} and token.pos_ in {"VERB", "AUX"}:
            left = max(sentence.start, token.left_edge.i)
            if left > sentence.start and sentence.doc[left - 1].is_punct:
                starts[left] = f"dependency_{token.dep_}"
        elif token.text in {";", "—", "–"} and token.i + 1 < sentence.end:
            starts[token.i + 1] = "strong_punctuation"
    return starts


def clause_spans(sentence: Span) -> list[tuple[Span, str]]:
    candidates = candidate_clause_starts(sentence)
    starts = [sentence.start, *sorted(candidates)]
    provisional: list[tuple[int, int, str]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else sentence.end
        rule = "sentence" if index == 0 else candidates[start]
        if start < end:
            provisional.append((start, end, rule))

    merged: list[tuple[int, int, str]] = []
    for start, end, rule in provisional:
        if alpha_word_count(list(sentence.doc[start:end])) < 3 and merged:
            previous_start, _, previous_rule = merged[-1]
            merged[-1] = (previous_start, end, previous_rule)
        else:
            merged.append((start, end, rule))
    if len(merged) > 1 and alpha_word_count(list(sentence.doc[merged[-1][0] : merged[-1][1]])) < 3:
        prior_start, _, prior_rule = merged[-2]
        merged[-2] = (prior_start, merged[-1][1], prior_rule)
        merged.pop()

    return [(sentence.doc[start:end], rule) for start, end, rule in merged]


def document_sentence_spans(parsed: Any, title_end_char: int) -> list[Span]:
    """Return parser sentences while always separating the title from the content."""
    starts = {sentence.start for sentence in parsed.sents}
    content_start = next(
        (
            token.i
            for token in parsed
            if token.idx >= title_end_char and not token.is_space
        ),
        None,
    )
    if content_start is not None:
        starts.add(content_start)

    ordered_starts = sorted(starts)
    spans: list[Span] = []
    for index, start in enumerate(ordered_starts):
        end = ordered_starts[index + 1] if index + 1 < len(ordered_starts) else len(parsed)
        span = parsed[start:end]
        if span.text.strip():
            spans.append(span)
    return spans


def trimmed_offsets(span: Span) -> tuple[int, int, str]:
    """Return offsets whose slice exactly equals the stored, whitespace-trimmed text."""
    raw_text = span.text
    left_trim = len(raw_text) - len(raw_text.lstrip())
    right_trim = len(raw_text) - len(raw_text.rstrip())
    start_char = span.start_char + left_trim
    end_char = span.end_char - right_trim
    return start_char, end_char, span.doc.text[start_char:end_char]


DOCUMENT_FIELDS = (
    "document_id", "source_id", "platform", "title", "published_at", "time_basis",
    "content_kind", "matched_queries", "source_path", "character_count", "sentence_count",
    "clause_count", "document_text",
)
SENTENCE_FIELDS = (
    "sentence_id", "document_id", "source_id", "platform", "sentence_index",
    "source_segment", "start_char", "end_char", "token_count", "sentence_text",
)
CLAUSE_FIELDS = (
    "clause_id", "sentence_id", "document_id", "source_id", "platform",
    "sentence_index", "clause_index", "source_segment", "split_rule", "start_char",
    "end_char", "token_count", "clause_text",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x-csv", type=Path, default=DEFAULT_X_CSV)
    parser.add_argument("--guardian-csv", type=Path, default=DEFAULT_GUARDIAN_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=MODEL_NAME)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    x_csv = args.x_csv.resolve()
    guardian_csv = args.guardian_csv.resolve()
    output_dir = args.output_dir.resolve()
    documents = load_documents(x_csv, guardian_csv)
    if not documents:
        raise SystemExit("No source documents were loaded")

    nlp: Language = spacy.load(args.model, disable=["ner"])
    max_chars = max(len(row["document_text"]) for row in documents)
    nlp.max_length = max(nlp.max_length, max_chars + 1_000)

    sentence_rows: list[dict[str, Any]] = []
    clause_rows: list[dict[str, Any]] = []
    processed_documents: list[dict[str, Any]] = []
    texts = (row["document_text"] for row in documents)
    for doc_index, (record, parsed) in enumerate(
        zip(documents, nlp.pipe(texts, batch_size=16), strict=True), start=1
    ):
        document_sentences = document_sentence_spans(parsed, record["title_end_char"])
        document_clause_count = 0
        for sentence_index, sentence in enumerate(document_sentences):
            sentence_id = f"{record['document_id']}::s{sentence_index:04d}"
            sentence_start_char, sentence_end_char, sentence_text = trimmed_offsets(sentence)
            segment = source_segment(
                sentence_start_char,
                sentence_end_char,
                record["title_end_char"],
                record["content_segment_name"],
            )
            sentence_rows.append(
                {
                    "sentence_id": sentence_id,
                    "document_id": record["document_id"],
                    "source_id": record["source_id"],
                    "platform": record["platform"],
                    "sentence_index": sentence_index,
                    "source_segment": segment,
                    "start_char": sentence_start_char,
                    "end_char": sentence_end_char,
                    "token_count": sum(not token.is_space for token in sentence),
                    "sentence_text": sentence_text,
                }
            )
            for clause_index, (clause, rule) in enumerate(clause_spans(sentence)):
                start_char, end_char, clause_text = trimmed_offsets(clause)
                clause_rows.append(
                    {
                        "clause_id": f"{sentence_id}::c{clause_index:03d}",
                        "sentence_id": sentence_id,
                        "document_id": record["document_id"],
                        "source_id": record["source_id"],
                        "platform": record["platform"],
                        "sentence_index": sentence_index,
                        "clause_index": clause_index,
                        "source_segment": source_segment(
                            start_char,
                            end_char,
                            record["title_end_char"],
                            record["content_segment_name"],
                        ),
                        "split_rule": rule,
                        "start_char": start_char,
                        "end_char": end_char,
                        "token_count": sum(not token.is_space for token in clause),
                        "clause_text": clause_text,
                    }
                )
                document_clause_count += 1

        processed_documents.append(
            {
                **record,
                "character_count": len(record["document_text"]),
                "sentence_count": len(document_sentences),
                "clause_count": document_clause_count,
            }
        )
        if doc_index % 100 == 0 or doc_index == len(documents):
            print(f"Processed {doc_index}/{len(documents)} documents")

    document_counts = Counter(row["platform"] for row in processed_documents)
    sentence_counts = Counter(row["platform"] for row in sentence_rows)
    clause_counts = Counter(row["platform"] for row in clause_rows)
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    input_details = {
        "x": {
            "source_csv": str(x_csv),
            "source_csv_sha256": sha256_file(x_csv),
        },
        "guardian": {
            "source_csv": str(guardian_csv),
            "source_csv_sha256": sha256_file(guardian_csv),
        },
    }
    platform_outputs: dict[str, dict[str, Any]] = {}
    for platform in ("x", "guardian"):
        platform_dir = output_dir / platform
        platform_documents = [row for row in processed_documents if row["platform"] == platform]
        platform_sentences = [row for row in sentence_rows if row["platform"] == platform]
        platform_clauses = [row for row in clause_rows if row["platform"] == platform]

        write_csv(platform_dir / "documents.csv", platform_documents, DOCUMENT_FIELDS)
        write_csv(platform_dir / "sentences.csv", platform_sentences, SENTENCE_FIELDS)
        write_csv(platform_dir / "clauses.csv", platform_clauses, CLAUSE_FIELDS)

        platform_manifest = {
            "created_at": created_at,
            "platform": platform,
            "spacy_model": args.model,
            "spacy_version": spacy.__version__,
            "split_rule_version": SPLIT_RULE_VERSION,
            "input": input_details[platform],
            "counts": {
                "documents": len(platform_documents),
                "sentences": len(platform_sentences),
                "clauses": len(platform_clauses),
            },
            "notes": [
                "X documents combine headline and summary."
                if platform == "x"
                else "Guardian documents combine title and full body HTML converted to plain text.",
                "Titles are forced to end before summary/body sentence segmentation begins.",
                "Guardian interactive atoms are excluded as non-article page furniture."
                if platform == "guardian"
                else "X News API summaries are used because full article bodies are unavailable.",
                "Clause boundaries are conservative dependency/punctuation rules and are non-overlapping.",
                "Character offsets are relative to this platform's documents.csv document_text.",
            ],
        }
        atomic_json(platform_dir / "preprocessing_manifest.json", platform_manifest)
        platform_outputs[platform] = {
            "directory": str(platform_dir),
            "documents": len(platform_documents),
            "sentences": len(platform_sentences),
            "clauses": len(platform_clauses),
        }

    manifest = {
        "created_at": created_at,
        "spacy_model": args.model,
        "spacy_version": spacy.__version__,
        "split_rule_version": SPLIT_RULE_VERSION,
        "inputs": input_details,
        "outputs": platform_outputs,
        "counts": {
            "documents": dict(document_counts),
            "sentences": dict(sentence_counts),
            "clauses": dict(clause_counts),
            "total_documents": len(processed_documents),
            "total_sentences": len(sentence_rows),
            "total_clauses": len(clause_rows),
        },
        "notes": [
            "X documents combine headline and summary.",
            "Guardian documents combine title and full body HTML converted to plain text.",
            "Titles are forced to end before summary/body sentence segmentation begins.",
            "Guardian interactive atoms are excluded as non-article page furniture.",
            "Clause boundaries are conservative dependency/punctuation rules and are non-overlapping.",
            "Character offsets are relative to documents.csv document_text.",
        ],
    }
    atomic_json(output_dir / "preprocessing_manifest.json", manifest)

    # Remove superseded mixed-platform tables only after both platform outputs succeed.
    for mixed_filename in ("documents.csv", "sentences.csv", "clauses.csv"):
        (output_dir / mixed_filename).unlink(missing_ok=True)

    print(
        f"Saved separate X and Guardian outputs: {len(processed_documents)} documents, "
        f"{len(sentence_rows)} sentences, and {len(clause_rows)} clauses under {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
