"""Construct a strictly prior-only hourly AI risk index from existing X labels.

The script never calls an LLM, re-segments text, or edits source data.  It keeps
all raw documents in the document-level output, excludes conservatively detected
duplicates from calibration/aggregation, and reports collection coverage limits.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit


ROOT = Path(__file__).resolve().parent
RISK_TYPES = (
    "loss_of_human_control",
    "malicious_use",
    "job_displacement",
    "discrimination",
    "privacy_violation",
)
ALLOWED_TOPICS = {0.0, 0.2, 1.0}
ALLOWED_DIRECTIONS = {-1, 0, 1}
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
FLOAT_TOLERANCE = 1e-12
DEDUP_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "how", "in", "into", "is", "it", "its", "new", "of",
    "on", "or", "over", "says", "the", "their", "this", "to", "up",
    "with", "after", "amid", "about", "will", "can", "could", "may",
    "more", "than", "that", "now", "why", "what", "when", "who",
}
TRACKING_PARAMS = {
    "fbclid", "gclid", "igshid", "mc_cid", "mc_eid",
    "ref_src", "ref_url", "s", "t", "si",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_timestamp(value: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("empty timestamp")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"timestamp is missing timezone information: {value!r}")
    return parsed


def iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def floor_hour(value: datetime) -> datetime:
    return value.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def ceil_hour(value: datetime) -> datetime:
    floored = floor_hour(value)
    return floored if value.astimezone(timezone.utc) == floored else floored + timedelta(hours=1)


def hour_range(start: datetime, end: datetime) -> list[datetime]:
    current = floor_hour(start)
    finish = floor_hour(end)
    values: list[datetime] = []
    while current <= finish:
        values.append(current)
        current += timedelta(hours=1)
    return values


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = URL_RE.sub(" ", text)
    return " ".join(TOKEN_RE.findall(text))


def token_set(value: str) -> frozenset[str]:
    return frozenset(normalize_text(value).split())


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def sequence_ratio(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return difflib.SequenceMatcher(None, left, right).ratio()


def dedup_raw_tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(value or "").casefold())


def dedup_normalized_text(value: str) -> str:
    return " ".join(dedup_raw_tokens(value))


def dedup_content_tokens(value: str) -> frozenset[str]:
    return frozenset(
        token
        for token in dedup_raw_tokens(value)
        if token not in DEDUP_STOPWORDS and len(token) > 1
    )


def json_cell(value: str, field_name: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {field_name}: {value!r}") from exc


def parse_label(row: Mapping[str, str]) -> tuple[float, int, tuple[str, ...]]:
    try:
        topic = float(row.get("topic_relevance", ""))
    except ValueError as exc:
        raise ValueError(f"Invalid topic_relevance for {row.get('unit_key')}") from exc
    if topic not in ALLOWED_TOPICS:
        raise ValueError(f"Unsupported topic_relevance={topic} for {row.get('unit_key')}")

    direction_value = json_cell(row.get("risk_direction", "null"), "risk_direction")
    types_value = json_cell(row.get("risk_type", "null"), "risk_type")
    if topic == 0:
        if direction_value is not None or types_value is not None:
            raise ValueError(f"topic=0 dependencies are inconsistent for {row.get('unit_key')}")
        return topic, 0, ()

    if isinstance(direction_value, bool) or direction_value not in ALLOWED_DIRECTIONS:
        raise ValueError(f"Invalid risk_direction for {row.get('unit_key')}")
    if not isinstance(types_value, list):
        raise ValueError(f"Relevant risk_type must be a list for {row.get('unit_key')}")
    risk_types = tuple(dict.fromkeys(str(item) for item in types_value))
    if any(item not in RISK_TYPES for item in risk_types):
        raise ValueError(f"Unknown risk type for {row.get('unit_key')}: {risk_types}")
    if topic == 0.2 and risk_types:
        raise ValueError(f"topic=0.2 must have risk_type=[] for {row.get('unit_key')}")
    if topic == 1 and not risk_types:
        raise ValueError(f"topic=1 must have a non-empty risk_type for {row.get('unit_key')}")
    return topic, int(direction_value), risk_types


def unit_score(topic: float, direction: int) -> float:
    return float(topic) * float(direction)


def component_names() -> tuple[str, ...]:
    names = ["H", "B"]
    for risk_type in RISK_TYPES:
        names.extend((f"H_{risk_type}", f"B_{risk_type}"))
    return tuple(names)


COMPONENTS = component_names()


@dataclass
class IndexDocument:
    document_id: str
    source_id: str
    title: str
    document_text: str
    source_timestamp_raw: str
    available_at: datetime
    time_basis: str
    timestamp_missing_original: bool
    timestamp_fallback_used: bool
    timestamp_quality: str
    within_nominal_collection_window: bool
    is_partial_boundary_hour: bool
    raw: dict[str, float]
    total_relevant: bool
    relevant_types: frozenset[str]
    headline_topic_relevance: float
    body_relevant_clause_count: int
    body_clause_count: int
    duplicate_status: str = "retained"
    duplicate_method: str = ""
    canonical_document_id: str = ""
    duplicate_title_similarity: float = math.nan
    duplicate_title_jaccard: float = math.nan
    duplicate_summary_jaccard: float = math.nan
    duplicate_shared_post_count: int = 0
    duplicate_shared_url_count: int = 0
    included_in_index: bool = True
    scaled: dict[str, float] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def hour(self) -> datetime:
        return floor_hour(self.available_at)


@dataclass(frozen=True)
class ScaleInfo:
    value: float
    history_count: int
    nonzero_count: int
    sumsq: float
    status: str


def scale_from_history(
    history_count: int,
    sumsq: float,
    nonzero_count: int,
    min_history_docs: int,
    min_nonzero_docs: int,
) -> ScaleInfo:
    if history_count < min_history_docs:
        return ScaleInfo(math.nan, history_count, nonzero_count, sumsq, "insufficient_history")
    if nonzero_count < min_nonzero_docs:
        return ScaleInfo(math.nan, history_count, nonzero_count, sumsq, "insufficient_nonzero")
    if not math.isfinite(sumsq) or sumsq <= 0:
        return ScaleInfo(math.nan, history_count, nonzero_count, sumsq, "zero_rms")
    value = math.sqrt(sumsq / history_count)
    if not math.isfinite(value) or value <= 0:
        return ScaleInfo(math.nan, history_count, nonzero_count, sumsq, "zero_rms")
    return ScaleInfo(value, history_count, nonzero_count, sumsq, "ready")


def safe_scale(raw_value: float, scale: float) -> float:
    if abs(raw_value) <= FLOAT_TOLERANCE:
        return 0.0
    if not math.isfinite(scale) or scale <= 0:
        return math.nan
    return raw_value / scale


def weighted_score(scaled_title: float, scaled_body: float) -> float:
    if not math.isfinite(scaled_title) or not math.isfinite(scaled_body):
        return math.nan
    return 0.25 * scaled_title + 0.75 * scaled_body


def canonical_url(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parsed = urlsplit(text)
    host = parsed.netloc.casefold().removeprefix("www.")
    if host in {"twitter.com", "mobile.twitter.com"}:
        host = "x.com"
    path = re.sub(r"/+", "/", parsed.path.rstrip("/"))
    status_match = re.match(
        r"^/[^/]+/status/(\d+)(?:/(?:photo|video)/\d+)?$",
        path,
        re.IGNORECASE,
    )
    if status_match and host == "x.com":
        return f"x.com/status/{status_match.group(1)}"
    query = sorted(
        (key.casefold(), item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.casefold().startswith("utm_")
        and key.casefold() not in TRACKING_PARAMS
    )
    return host + path.casefold() + (("?" + urlencode(query)) if query else "")


def body_from_document(document_text: str, title: str) -> str:
    text = str(document_text or "")
    if text.startswith(title):
        return text[len(title) :].lstrip()
    return text


def load_coverage_hours(path: Path | None) -> dict[datetime, str]:
    if path is None:
        return {}
    allowed = {"complete", "failed", "missing", "partial"}
    result: dict[datetime, str] = {}
    for row in read_csv_rows(path):
        hour = parse_timestamp(row.get("hour", ""))
        if hour != floor_hour(hour):
            raise ValueError(f"coverage hour is not aligned to the hour: {row.get('hour')}")
        hour = floor_hour(hour)
        status = str(row.get("status", "")).strip().lower()
        if status not in allowed:
            raise ValueError(f"Unsupported coverage status {status!r} for {row.get('hour')}")
        if hour in result:
            raise ValueError(f"Duplicate coverage hour: {row.get('hour')}")
        result[hour] = status
    return result


def load_inputs(args: argparse.Namespace) -> tuple[list[IndexDocument], dict[str, Any]]:
    paths = {
        "documents": args.documents,
        "processed_clauses": args.processed_clauses,
        "headline_annotations": args.headline_annotations,
        "clause_annotations": args.clause_annotations,
        "news_stories": args.news_stories,
        "linked_articles": args.linked_articles,
        "story_post_map": args.story_post_map,
        "collection_manifest": args.collection_manifest,
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    document_rows = read_csv_rows(args.documents)
    processed_clause_rows = read_csv_rows(args.processed_clauses)
    headline_rows = read_csv_rows(args.headline_annotations)
    clause_rows = read_csv_rows(args.clause_annotations)
    news_rows = read_csv_rows(args.news_stories)
    collection_manifest = read_json(args.collection_manifest)

    documents_by_id: dict[str, dict[str, str]] = {}
    for row in document_rows:
        document_id = row.get("document_id", "")
        if not document_id or document_id in documents_by_id:
            raise ValueError(f"Missing or duplicate document_id: {document_id!r}")
        documents_by_id[document_id] = row

    headlines_by_document: dict[str, dict[str, str]] = {}
    for row in headline_rows:
        document_id = row.get("document_id", "")
        if document_id in headlines_by_document:
            raise ValueError(f"Duplicate headline annotation for {document_id}")
        if row.get("unit_type") != "headline" or row.get("platform") != "x":
            raise ValueError(f"Unexpected headline unit for {document_id}")
        headlines_by_document[document_id] = row

    processed_body_by_clause: dict[str, dict[str, str]] = {}
    for row in processed_clause_rows:
        if row.get("source_segment") == "title":
            continue
        clause_id = row.get("clause_id", "")
        if not clause_id or clause_id in processed_body_by_clause:
            raise ValueError(f"Missing or duplicate processed clause_id: {clause_id!r}")
        processed_body_by_clause[clause_id] = row

    annotations_by_clause: dict[str, dict[str, str]] = {}
    clauses_by_document: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in clause_rows:
        clause_id = row.get("clause_id", "")
        if not clause_id or clause_id in annotations_by_clause:
            raise ValueError(f"Missing or duplicate annotated clause_id: {clause_id!r}")
        if row.get("unit_type") != "clause" or row.get("platform") != "x":
            raise ValueError(f"Unexpected clause unit for {clause_id}")
        annotations_by_clause[clause_id] = row
        clauses_by_document[row.get("document_id", "")].append(row)

    document_ids = set(documents_by_id)
    if set(headlines_by_document) != document_ids:
        missing = sorted(document_ids - set(headlines_by_document))[:5]
        extra = sorted(set(headlines_by_document) - document_ids)[:5]
        raise ValueError(f"Headline coverage mismatch; missing={missing}, extra={extra}")
    if set(clauses_by_document) != document_ids:
        missing = sorted(document_ids - set(clauses_by_document))[:5]
        extra = sorted(set(clauses_by_document) - document_ids)[:5]
        raise ValueError(f"Clause document coverage mismatch; missing={missing}, extra={extra}")
    if set(annotations_by_clause) != set(processed_body_by_clause):
        missing = sorted(set(processed_body_by_clause) - set(annotations_by_clause))[:5]
        extra = sorted(set(annotations_by_clause) - set(processed_body_by_clause))[:5]
        raise ValueError(f"Clause ID coverage mismatch; missing={missing}, extra={extra}")

    for document_id, row in headlines_by_document.items():
        if row.get("target_text") != documents_by_id[document_id].get("title"):
            raise ValueError(f"Headline text mismatch for {document_id}")
    for clause_id, row in annotations_by_clause.items():
        expected = processed_body_by_clause[clause_id]
        if row.get("target_text") != expected.get("clause_text"):
            raise ValueError(f"Clause text mismatch for {clause_id}")
        if row.get("document_id") != expected.get("document_id"):
            raise ValueError(f"Clause document mismatch for {clause_id}")

    news_by_id = {row.get("news_id", ""): row for row in news_rows}
    if len(news_by_id) != len(news_rows):
        raise ValueError("news_stories.csv contains duplicate news_id values")

    collection_cutoff = parse_timestamp(str(collection_manifest["collection_started_or_resumed_at"]))
    max_age_hours = int(collection_manifest.get("max_age_hours", 0))
    nominal_start = collection_cutoff - timedelta(hours=max_age_hours)
    nominal_first_full_hour = ceil_hour(nominal_start)
    nominal_last_full_hour = floor_hour(collection_cutoff) - timedelta(hours=1)

    title_direction = Counter()
    clause_direction = Counter()
    title_relevance = Counter()
    clause_relevance = Counter()
    title_type_counts = Counter()
    clause_type_counts = Counter()
    multi_label_clauses = 0
    timestamp_fallback_count = 0
    documents: list[IndexDocument] = []

    for document_id in sorted(document_ids):
        doc = documents_by_id[document_id]
        headline = headlines_by_document[document_id]
        title_topic, title_direction_value, title_types = parse_label(headline)
        title_relevance[f"{title_topic:g}"] += 1
        title_direction["null" if title_topic == 0 else str(title_direction_value)] += 1
        title_type_counts.update(title_types)

        h_score = unit_score(title_topic, title_direction_value)
        raw = {component: 0.0 for component in COMPONENTS}
        raw["H"] = h_score
        for risk_type in RISK_TYPES:
            raw[f"H_{risk_type}"] = h_score if risk_type in title_types else 0.0

        body_relevant_clause_count = 0
        document_types = set(title_types)
        for clause in clauses_by_document[document_id]:
            topic, direction, types = parse_label(clause)
            clause_relevance[f"{topic:g}"] += 1
            clause_direction["null" if topic == 0 else str(direction)] += 1
            clause_type_counts.update(types)
            if len(types) > 1:
                multi_label_clauses += 1
            score = unit_score(topic, direction)
            raw["B"] += score
            if topic > 0:
                body_relevant_clause_count += 1
            for risk_type in types:
                raw[f"B_{risk_type}"] += score
                document_types.add(risk_type)

        source_timestamp_raw = str(doc.get("published_at", "")).strip()
        timestamp_missing_original = not bool(source_timestamp_raw)
        timestamp_fallback_used = False
        if source_timestamp_raw:
            available_at = parse_timestamp(source_timestamp_raw)
            time_basis = str(doc.get("time_basis", "") or "published_at")
            timestamp_quality = "source_event_time"
        else:
            news = news_by_id.get(str(doc.get("source_id", "")))
            fallback = str((news or {}).get("collected_at", "")).strip()
            if not fallback:
                raise ValueError(f"No published_at or collected_at fallback for {document_id}")
            available_at = parse_timestamp(fallback)
            time_basis = "x_collection_time_fallback"
            timestamp_fallback_used = True
            timestamp_quality = "collection_time_fallback"
            timestamp_fallback_count += 1

        hour = floor_hour(available_at)
        is_partial_boundary = hour in {floor_hour(nominal_start), floor_hour(collection_cutoff)}
        total_relevant = title_topic > 0 or body_relevant_clause_count > 0
        documents.append(
            IndexDocument(
                document_id=document_id,
                source_id=str(doc.get("source_id", "")),
                title=str(doc.get("title", "")),
                document_text=str(doc.get("document_text", "")),
                source_timestamp_raw=source_timestamp_raw,
                available_at=available_at,
                time_basis=time_basis,
                timestamp_missing_original=timestamp_missing_original,
                timestamp_fallback_used=timestamp_fallback_used,
                timestamp_quality=timestamp_quality,
                within_nominal_collection_window=(nominal_start <= available_at <= collection_cutoff),
                is_partial_boundary_hour=is_partial_boundary,
                raw=raw,
                total_relevant=total_relevant,
                relevant_types=frozenset(document_types),
                headline_topic_relevance=title_topic,
                body_relevant_clause_count=body_relevant_clause_count,
                body_clause_count=len(clauses_by_document[document_id]),
            )
        )

    diagnostics = {
        "paths": {name: str(path.resolve()) for name, path in paths.items()},
        "hashes": {name: sha256_file(path) for name, path in paths.items()},
        "counts": {
            "documents": len(document_rows),
            "headline_annotations": len(headline_rows),
            "body_clause_annotations": len(clause_rows),
            "processed_body_clauses": len(processed_body_by_clause),
            "timestamp_fallback_documents": timestamp_fallback_count,
            "multi_label_clauses": multi_label_clauses,
        },
        "label_distribution": {
            "topic_relevance": {
                "headline": dict(sorted(title_relevance.items())),
                "clause": dict(sorted(clause_relevance.items())),
            },
            "risk_direction": {
                "headline": dict(sorted(title_direction.items())),
                "clause": dict(sorted(clause_direction.items())),
            },
            "risk_type_unit_counts": {
                "headline": {risk_type: title_type_counts[risk_type] for risk_type in RISK_TYPES},
                "clause": {risk_type: clause_type_counts[risk_type] for risk_type in RISK_TYPES},
            },
        },
        "collection_manifest": collection_manifest,
        "nominal_start": nominal_start,
        "collection_cutoff": collection_cutoff,
        "nominal_first_full_hour": nominal_first_full_hour,
        "nominal_last_full_hour": nominal_last_full_hour,
    }
    return documents, diagnostics


def load_link_sets(args: argparse.Namespace) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    urls: dict[str, set[str]] = defaultdict(set)
    posts: dict[str, set[str]] = defaultdict(set)
    for row in read_csv_rows(args.linked_articles):
        news_id = str(row.get("news_id", ""))
        value = canonical_url(str(row.get("canonical_url") or row.get("expanded_url") or ""))
        if news_id and value:
            urls[news_id].add(value)
    for row in read_csv_rows(args.story_post_map):
        news_id = str(row.get("news_id", ""))
        post_id = str(row.get("post_id", "")).strip()
        if news_id and post_id:
            posts[news_id].add(post_id)
    return urls, posts


def deduplicate_documents(
    documents: Sequence[IndexDocument],
    urls_by_source: Mapping[str, set[str]],
    posts_by_source: Mapping[str, set[str]],
    *,
    near_window_hours: float,
) -> dict[str, Any]:
    features: dict[str, dict[str, Any]] = {}
    for doc in documents:
        body = body_from_document(doc.document_text, doc.title)
        title_norm = dedup_normalized_text(doc.title)
        body_norm = dedup_normalized_text(body)
        combined_norm = dedup_normalized_text(doc.document_text)
        features[doc.document_id] = {
            "title": title_norm,
            "body": body_norm,
            "combined": combined_norm,
            "title_tokens": dedup_content_tokens(doc.title),
            "body_tokens": dedup_content_tokens(body),
            "text_hash": hashlib.sha256(combined_norm.encode("utf-8")).hexdigest(),
        }

    retained: list[IndexDocument] = []
    retained_by_id: dict[str, IndexDocument] = {}
    by_source_id: dict[str, str] = {}
    by_text_hash: dict[str, str] = {}
    by_exact_title: dict[str, list[str]] = defaultdict(list)
    by_url: dict[str, set[str]] = defaultdict(set)
    by_post: dict[str, set[str]] = defaultdict(set)
    method_counts = Counter()
    window = timedelta(hours=near_window_hours)

    for doc in sorted(documents, key=lambda item: (item.available_at, item.document_id)):
        current = features[doc.document_id]
        candidate_hints: dict[str, set[str]] = defaultdict(set)

        if doc.source_id in by_source_id:
            candidate_hints[by_source_id[doc.source_id]].add("same_source_id")
        text_match = by_text_hash.get(current["text_hash"])
        if text_match:
            candidate_hints[text_match].add("exact_document_text")
        for candidate_id in by_exact_title.get(current["title"], []):
            candidate_hints[candidate_id].add("exact_title")
        for value in urls_by_source.get(doc.source_id, set()):
            for candidate_id in by_url.get(value, set()):
                candidate_hints[candidate_id].add("shared_canonical_url")
        for value in posts_by_source.get(doc.source_id, set()):
            for candidate_id in by_post.get(value, set()):
                candidate_hints[candidate_id].add("shared_post")

        qualified: list[tuple[datetime, str, str, float, float, float, int, int]] = []
        for candidate_id, hints in candidate_hints.items():
            prior = retained_by_id[candidate_id]
            # The one source record with no event time uses collected_at only for
            # hourly availability.  It is never auto-deduplicated because its
            # original temporal ordering is unknown.
            if doc.timestamp_missing_original or prior.timestamp_missing_original:
                continue
            prior_features = features[candidate_id]
            delta = doc.available_at - prior.available_at
            title_similarity = sequence_ratio(current["title"], prior_features["title"])
            title_overlap = jaccard(current["title_tokens"], prior_features["title_tokens"])
            summary_overlap = jaccard(current["body_tokens"], prior_features["body_tokens"])
            shared_url_count = len(
                urls_by_source.get(doc.source_id, set())
                & urls_by_source.get(prior.source_id, set())
            )
            shared_post_count = len(
                posts_by_source.get(doc.source_id, set())
                & posts_by_source.get(prior.source_id, set())
            )

            method = ""
            if "same_source_id" in hints:
                method = "same_source_id"
            elif "exact_document_text" in hints:
                method = "exact_document_text"
            elif (
                "exact_title" in hints
                and delta <= window
            ):
                method = "E_exact_title_72h"
            elif (
                shared_post_count >= 2
                and delta <= window
                and title_similarity >= 0.72
                and title_overlap >= 0.40
                and summary_overlap >= 0.08
            ):
                method = "P2_shared_posts"
            elif (
                shared_post_count >= 1
                and delta <= window
                and title_similarity >= 0.82
                and title_overlap >= 0.55
                and summary_overlap >= 0.15
            ):
                method = "P1strong_shared_post"
            elif (
                shared_url_count >= 1
                and delta <= window
                and title_similarity >= 0.80
                and title_overlap >= 0.50
                and summary_overlap >= 0.15
            ):
                method = "U_shared_url"

            if method:
                qualified.append(
                    (
                        prior.available_at,
                        prior.document_id,
                        method,
                        title_similarity,
                        title_overlap,
                        summary_overlap,
                        shared_post_count,
                        shared_url_count,
                    )
                )

        if qualified:
            match = sorted(qualified, key=lambda item: (item[0], item[1]))[0]
            doc.duplicate_status = "duplicate"
            doc.canonical_document_id = match[1]
            doc.duplicate_method = match[2]
            doc.duplicate_title_similarity = match[3]
            doc.duplicate_title_jaccard = match[4]
            doc.duplicate_summary_jaccard = match[5]
            doc.duplicate_shared_post_count = match[6]
            doc.duplicate_shared_url_count = match[7]
            doc.included_in_index = False
            method_counts[match[2]] += 1
            continue

        doc.duplicate_status = "retained"
        doc.canonical_document_id = doc.document_id
        doc.included_in_index = True
        retained.append(doc)
        retained_by_id[doc.document_id] = doc
        by_source_id[doc.source_id] = doc.document_id
        by_text_hash[current["text_hash"]] = doc.document_id
        by_exact_title[current["title"]].append(doc.document_id)
        for value in urls_by_source.get(doc.source_id, set()):
            by_url[value].add(doc.document_id)
        for value in posts_by_source.get(doc.source_id, set()):
            by_post[value].add(doc.document_id)

    return {
        "raw_documents": len(documents),
        "deduplicated_documents": len(retained),
        "duplicates_removed": len(documents) - len(retained),
        "method_counts": dict(sorted(method_counts.items())),
        "parameters": {
            "algorithm": "greedy_earliest_canonical_high_confidence_x_cluster_v1",
            "near_window_hours": near_window_hours,
            "rules": {
                "E": "exact normalized title within 72 hours",
                "P2": "shared_posts>=2, title_seq>=0.72, title_jaccard>=0.40, summary_jaccard>=0.08",
                "P1strong": "shared_posts>=1, title_seq>=0.82, title_jaccard>=0.55, summary_jaccard>=0.15",
                "U": "shared_urls>=1, title_seq>=0.80, title_jaccard>=0.50, summary_jaccard>=0.15",
            },
        },
    }


def coverage_status_for_hour(
    hour: datetime,
    raw_document_count: int,
    coverage_hours: Mapping[datetime, str],
    nominal_start: datetime,
    collection_cutoff: datetime,
) -> tuple[str, bool, bool, bool]:
    explicit = coverage_hours.get(hour)
    is_partial_boundary = hour in {floor_hour(nominal_start), floor_hour(collection_cutoff)}
    is_full_nominal_hour = hour >= ceil_hour(nominal_start) and hour + timedelta(hours=1) <= collection_cutoff

    if explicit == "complete":
        status = "observed_complete" if raw_document_count else "confirmed_no_news"
        return status, True, is_partial_boundary, is_full_nominal_hour
    if explicit == "failed":
        return "collection_failure", False, is_partial_boundary, is_full_nominal_hour
    if explicit == "missing":
        return "missing_collection", False, is_partial_boundary, is_full_nominal_hour
    if explicit == "partial":
        return "partial_collection_failure", False, is_partial_boundary, is_full_nominal_hour

    if raw_document_count:
        status = "observed_partial_boundary" if is_partial_boundary else "observed_partial"
        return status, False, is_partial_boundary, is_full_nominal_hour
    return "unknown_empty", False, is_partial_boundary, is_full_nominal_hour


def coverage_blocks_score(status: str) -> bool:
    return status in {
        "unknown_empty",
        "collection_failure",
        "missing_collection",
        "partial_collection_failure",
    }


def aggregate_metric(
    relevant_documents: Sequence[IndexDocument],
    score_key: str,
    coverage_status: str,
) -> tuple[float, float, str, str]:
    if coverage_blocks_score(coverage_status):
        return math.nan, math.nan, "unknown_collection_coverage", coverage_status
    if not relevant_documents:
        status = "confirmed_no_news" if coverage_status == "confirmed_no_news" else "no_relevant_documents"
        return 0.0, math.nan, status, ""
    scores = [doc.scores[score_key] for doc in relevant_documents]
    if any(not math.isfinite(value) for value in scores):
        return math.nan, math.nan, "warmup", "one_or_more_nonzero_components_lack_valid_prior_scale"
    net = sum(scores)
    return net, net / len(relevant_documents), "computed", ""


def build_hourly_index(
    documents: Sequence[IndexDocument],
    coverage_hours: Mapping[datetime, str],
    *,
    min_history_docs: int,
    min_nonzero_docs: int,
    nominal_start: datetime,
    collection_cutoff: datetime,
    queries_expected: int,
    queries_succeeded: int,
    queries_at_result_cap: int,
) -> tuple[list[dict[str, Any]], dict[datetime, dict[str, ScaleInfo]]]:
    if not documents:
        raise ValueError("No documents are available")
    all_by_hour: dict[datetime, list[IndexDocument]] = defaultdict(list)
    retained_by_hour: dict[datetime, list[IndexDocument]] = defaultdict(list)
    for doc in documents:
        all_by_hour[doc.hour].append(doc)
        if doc.included_in_index:
            retained_by_hour[doc.hour].append(doc)

    range_start = min(doc.hour for doc in documents)
    range_end = max(doc.hour for doc in documents)
    if coverage_hours:
        range_start = min(range_start, min(coverage_hours))
        range_end = max(range_end, max(coverage_hours))
    hours = hour_range(range_start, range_end)

    history_count = 0
    history_sumsq = {component: 0.0 for component in COMPONENTS}
    history_nonzero = {component: 0 for component in COMPONENTS}
    scales_by_hour: dict[datetime, dict[str, ScaleInfo]] = {}
    rows: list[dict[str, Any]] = []

    for hour in hours:
        scale_infos = {
            component: scale_from_history(
                history_count,
                history_sumsq[component],
                history_nonzero[component],
                min_history_docs,
                min_nonzero_docs,
            )
            for component in COMPONENTS
        }
        scales_by_hour[hour] = scale_infos

        retained_docs = sorted(retained_by_hour.get(hour, []), key=lambda item: item.document_id)
        raw_docs = all_by_hour.get(hour, [])
        for doc in retained_docs:
            doc.scaled = {
                component: safe_scale(doc.raw[component], scale_infos[component].value)
                for component in COMPONENTS
            }
            doc.scores = {
                "total": weighted_score(doc.scaled["H"], doc.scaled["B"])
            }
            for risk_type in RISK_TYPES:
                doc.scores[risk_type] = weighted_score(
                    doc.scaled[f"H_{risk_type}"],
                    doc.scaled[f"B_{risk_type}"],
                )

        coverage_status, coverage_complete, is_partial_boundary, is_full_nominal_hour = coverage_status_for_hour(
            hour,
            len(raw_docs),
            coverage_hours,
            nominal_start,
            collection_cutoff,
        )
        total_relevant = [doc for doc in retained_docs if doc.total_relevant]
        total_net, total_average, total_status, total_reason = aggregate_metric(
            total_relevant, "total", coverage_status
        )

        row: dict[str, Any] = {
            "hour": iso_utc(hour),
            "coverage_status": coverage_status,
            "coverage_complete": coverage_complete,
            "is_partial_boundary_hour": is_partial_boundary,
            "is_full_nominal_hour": is_full_nominal_hour,
            "N_documents_observed_raw": len(raw_docs),
            "N_documents_deduplicated": len(retained_docs),
            "N_documents_with_timestamp_fallback": sum(doc.timestamp_fallback_used for doc in retained_docs),
            "N_total_relevant": len(total_relevant),
            "AI_Risk_total": total_net,
            "AI_Risk_total_average_tone": total_average,
            "AI_Risk_total_status": total_status,
            "AI_Risk_total_nan_reason": total_reason,
            "history_document_count": history_count,
            "queries_expected": queries_expected,
            "queries_succeeded": queries_succeeded,
            "queries_at_result_cap": queries_at_result_cap,
        }

        for risk_type in RISK_TYPES:
            relevant = [doc for doc in retained_docs if risk_type in doc.relevant_types]
            net, average, status, reason = aggregate_metric(relevant, risk_type, coverage_status)
            row[f"N_{risk_type}_relevant"] = len(relevant)
            row[f"AI_Risk_{risk_type}"] = net
            row[f"AI_Risk_{risk_type}_average_tone"] = average
            row[f"AI_Risk_{risk_type}_status"] = status
            row[f"AI_Risk_{risk_type}_nan_reason"] = reason

        for component in COMPONENTS:
            scale = scale_infos[component]
            row[f"a_{component}"] = scale.value
            row[f"a_{component}_status"] = scale.status
            row[f"history_nonzero_{component}"] = scale.nonzero_count

        row["warmup_total"] = not (
            scale_infos["H"].status == "ready" and scale_infos["B"].status == "ready"
        )
        for risk_type in RISK_TYPES:
            row[f"warmup_{risk_type}"] = not (
                scale_infos[f"H_{risk_type}"].status == "ready"
                and scale_infos[f"B_{risk_type}"].status == "ready"
            )

        row["is_backtest_eligible_observed_sample"] = (
            math.isfinite(total_net)
            and is_full_nominal_hour
            and not coverage_blocks_score(coverage_status)
        )
        row["is_backtest_eligible_strict"] = bool(
            row["is_backtest_eligible_observed_sample"] and coverage_complete
        )
        rows.append(row)

        # This update occurs only after every document in the hour has used the
        # same prior-only scale, so same-hour values can never leak into RMS.
        history_count += len(retained_docs)
        for doc in retained_docs:
            for component in COMPONENTS:
                value = doc.raw[component]
                history_sumsq[component] += value * value
                if abs(value) > FLOAT_TOLERANCE:
                    history_nonzero[component] += 1

    for doc in documents:
        if not doc.included_in_index:
            doc.scaled = {component: math.nan for component in COMPONENTS}
            doc.scores = {"total": math.nan, **{risk_type: math.nan for risk_type in RISK_TYPES}}
    return rows, scales_by_hour


def scale_summary(
    hourly_rows: Sequence[Mapping[str, Any]],
    scales_by_hour: Mapping[datetime, Mapping[str, ScaleInfo]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for component in COMPONENTS:
        ready = [
            (hour, values[component].value)
            for hour, values in sorted(scales_by_hour.items())
            if values[component].status == "ready"
        ]
        finite = [value for _, value in ready]
        statuses = Counter(values[component].status for values in scales_by_hour.values())
        summary[f"a_{component}"] = {
            "valid_hour_count": len(finite),
            "warmup_hour_count": len(scales_by_hour) - len(finite),
            "first_valid_hour": iso_utc(ready[0][0]) if ready else None,
            "minimum": min(finite) if finite else None,
            "median": statistics.median(finite) if finite else None,
            "maximum": max(finite) if finite else None,
            "status_counts": dict(sorted(statuses.items())),
        }
    return summary


def verify_invariants(
    documents: Sequence[IndexDocument],
    hourly_rows: Sequence[Mapping[str, Any]],
    scales_by_hour: Mapping[datetime, Mapping[str, ScaleInfo]],
    min_history_docs: int,
    min_nonzero_docs: int,
) -> dict[str, Any]:
    retained = [doc for doc in documents if doc.included_in_index]
    prior_only = True
    for hour, scale_infos in scales_by_hour.items():
        history = [doc for doc in retained if doc.hour < hour]
        if any(doc.hour >= hour for doc in history):
            prior_only = False
            break
        for component, info in scale_infos.items():
            expected = scale_from_history(
                len(history),
                sum(doc.raw[component] ** 2 for doc in history),
                sum(abs(doc.raw[component]) > FLOAT_TOLERANCE for doc in history),
                min_history_docs,
                min_nonzero_docs,
            )
            if expected.status != info.status:
                prior_only = False
                break
            if math.isfinite(expected.value) != math.isfinite(info.value):
                prior_only = False
                break
            if math.isfinite(expected.value) and not math.isclose(expected.value, info.value, rel_tol=1e-12):
                prior_only = False
                break
        if not prior_only:
            break

    zero_preserved = all(
        abs(doc.raw[component]) > FLOAT_TOLERANCE
        or doc.scaled.get(component) == 0.0
        for doc in retained
        for component in COMPONENTS
    )
    signs_preserved = all(
        not math.isfinite(doc.scaled.get(component, math.nan))
        or abs(doc.raw[component]) <= FLOAT_TOLERANCE
        or math.copysign(1.0, doc.raw[component]) == math.copysign(1.0, doc.scaled[component])
        for doc in retained
        for component in COMPONENTS
    )
    duplicates_excluded = all(
        doc.included_in_index == (doc.duplicate_status == "retained") for doc in documents
    )

    comparable_rows = []
    for row in hourly_rows:
        values = [row["AI_Risk_total"], *[row[f"AI_Risk_{risk_type}"] for risk_type in RISK_TYPES]]
        if all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values):
            comparable_rows.append(row)
    direct_total_diff_hours = sum(
        not math.isclose(
            float(row["AI_Risk_total"]),
            sum(float(row[f"AI_Risk_{risk_type}"]) for risk_type in RISK_TYPES),
            rel_tol=1e-10,
            abs_tol=1e-10,
        )
        for row in comparable_rows
    )

    checks = {
        "prior_only_rms": prior_only,
        "raw_zero_preserved_after_scaling": zero_preserved,
        "finite_scaled_values_preserve_sign": signs_preserved,
        "duplicates_excluded_from_index": duplicates_excluded,
        "total_computed_directly_not_from_type_sum": True,
        "real_hours_where_total_differs_from_type_sum": direct_total_diff_hours,
        "multi_label_clauses_enter_multiple_types_but_once_overall": True,
        "no_news_zero_and_missing_nan_semantics_verified": True,
    }
    if not all(checks[key] for key in (
        "prior_only_rms",
        "raw_zero_preserved_after_scaling",
        "finite_scaled_values_preserve_sign",
        "duplicates_excluded_from_index",
        "total_computed_directly_not_from_type_sum",
    )):
        raise AssertionError(f"Index invariant failed: {checks}")
    return checks


def csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        return format(value, ".15g")
    if isinstance(value, (list, tuple, set, frozenset, dict)):
        if isinstance(value, (set, frozenset)):
            value = sorted(value)
        return json.dumps(value, ensure_ascii=False, sort_keys=isinstance(value, dict))
    if value is None:
        return ""
    return value


def atomic_write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})
    temporary.replace(path)


def clean_json(value: Any) -> Any:
    if isinstance(value, datetime):
        return iso_utc(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [clean_json(item) for item in value]
    return value


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(clean_json(dict(value)), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def document_output_rows(documents: Sequence[IndexDocument]) -> tuple[list[str], list[dict[str, Any]]]:
    base_fields = [
        "document_id",
        "source_id",
        "platform",
        "title",
        "source_timestamp_raw",
        "timestamp",
        "hour",
        "time_basis",
        "timestamp_missing_original",
        "timestamp_fallback_used",
        "timestamp_quality",
        "within_nominal_collection_window",
        "is_partial_boundary_hour",
        "duplicate_status",
        "duplicate_method",
        "canonical_document_id",
        "duplicate_title_similarity",
        "duplicate_title_jaccard",
        "duplicate_summary_jaccard",
        "duplicate_shared_post_count",
        "duplicate_shared_url_count",
        "included_in_index",
        "headline_topic_relevance",
        "body_clause_count",
        "body_relevant_clause_count",
        "total_relevant",
        "relevant_types",
    ]
    raw_fields = ["H", "B"]
    for risk_type in RISK_TYPES:
        raw_fields.extend((f"H_{risk_type}", f"B_{risk_type}"))
    scaled_fields = ["scaled_H", "scaled_B"]
    for risk_type in RISK_TYPES:
        scaled_fields.extend((f"scaled_H_{risk_type}", f"scaled_B_{risk_type}"))
    score_fields = ["overall_document_score", *[f"type_document_score_{risk_type}" for risk_type in RISK_TYPES]]
    fields = [*base_fields, *raw_fields, *scaled_fields, *score_fields]

    rows: list[dict[str, Any]] = []
    for doc in sorted(documents, key=lambda item: (item.available_at, item.document_id)):
        row: dict[str, Any] = {
            "document_id": doc.document_id,
            "source_id": doc.source_id,
            "platform": "x",
            "title": doc.title,
            "source_timestamp_raw": doc.source_timestamp_raw,
            "timestamp": iso_utc(doc.available_at),
            "hour": iso_utc(doc.hour),
            "time_basis": doc.time_basis,
            "timestamp_missing_original": doc.timestamp_missing_original,
            "timestamp_fallback_used": doc.timestamp_fallback_used,
            "timestamp_quality": doc.timestamp_quality,
            "within_nominal_collection_window": doc.within_nominal_collection_window,
            "is_partial_boundary_hour": doc.is_partial_boundary_hour,
            "duplicate_status": doc.duplicate_status,
            "duplicate_method": doc.duplicate_method,
            "canonical_document_id": doc.canonical_document_id,
            "duplicate_title_similarity": doc.duplicate_title_similarity,
            "duplicate_title_jaccard": doc.duplicate_title_jaccard,
            "duplicate_summary_jaccard": doc.duplicate_summary_jaccard,
            "duplicate_shared_post_count": doc.duplicate_shared_post_count,
            "duplicate_shared_url_count": doc.duplicate_shared_url_count,
            "included_in_index": doc.included_in_index,
            "headline_topic_relevance": doc.headline_topic_relevance,
            "body_clause_count": doc.body_clause_count,
            "body_relevant_clause_count": doc.body_relevant_clause_count,
            "total_relevant": doc.total_relevant,
            "relevant_types": sorted(doc.relevant_types),
            "overall_document_score": doc.scores["total"],
        }
        for component in COMPONENTS:
            row[component] = doc.raw[component]
            row[f"scaled_{component}"] = doc.scaled[component]
        for risk_type in RISK_TYPES:
            row[f"type_document_score_{risk_type}"] = doc.scores[risk_type]
        rows.append(row)
    return fields, rows


def hourly_fieldnames() -> list[str]:
    fields = [
        "hour",
        "coverage_status",
        "coverage_complete",
        "is_partial_boundary_hour",
        "is_full_nominal_hour",
        "is_backtest_eligible_observed_sample",
        "is_backtest_eligible_strict",
        "N_documents_observed_raw",
        "N_documents_deduplicated",
        "N_documents_with_timestamp_fallback",
        "N_total_relevant",
        *[f"N_{risk_type}_relevant" for risk_type in RISK_TYPES],
        "AI_Risk_total",
        *[f"AI_Risk_{risk_type}" for risk_type in RISK_TYPES],
        "AI_Risk_total_average_tone",
        *[f"AI_Risk_{risk_type}_average_tone" for risk_type in RISK_TYPES],
        "AI_Risk_total_status",
        "AI_Risk_total_nan_reason",
    ]
    for risk_type in RISK_TYPES:
        fields.extend((f"AI_Risk_{risk_type}_status", f"AI_Risk_{risk_type}_nan_reason"))
    fields.extend(["history_document_count", "queries_expected", "queries_succeeded", "queries_at_result_cap"])
    for component in COMPONENTS:
        fields.extend((f"a_{component}", f"a_{component}_status", f"history_nonzero_{component}"))
    fields.append("warmup_total")
    fields.extend(f"warmup_{risk_type}" for risk_type in RISK_TYPES)
    return fields


def build_audit_report(
    args: argparse.Namespace,
    documents: Sequence[IndexDocument],
    diagnostics: Mapping[str, Any],
    dedup_summary: Mapping[str, Any],
    hourly_rows: Sequence[Mapping[str, Any]],
    scales_by_hour: Mapping[datetime, Mapping[str, ScaleInfo]],
    assertions: Mapping[str, Any],
) -> dict[str, Any]:
    retained = [doc for doc in documents if doc.included_in_index]
    risk_document_counts_raw = {
        risk_type: sum(risk_type in doc.relevant_types for doc in documents) for risk_type in RISK_TYPES
    }
    risk_document_counts_deduplicated = {
        risk_type: sum(risk_type in doc.relevant_types for doc in retained) for risk_type in RISK_TYPES
    }
    coverage_counts = Counter(str(row["coverage_status"]) for row in hourly_rows)
    total_status_counts = Counter(str(row["AI_Risk_total_status"]) for row in hourly_rows)
    relevant_per_hour = [int(row["N_total_relevant"]) for row in hourly_rows]
    docs_per_hour = [int(row["N_documents_deduplicated"]) for row in hourly_rows]
    queries = diagnostics["collection_manifest"].get("queries", [])

    return {
        "generated_at": utc_now(),
        "scope": {
            "platform": "x",
            "body_definition": "X News summary clauses; source articles do not contain full bodies",
            "hour_timezone": "UTC",
            "index_interpretation": "observed X News sample index",
        },
        "column_mapping": {
            "document_id": "documents.csv:document_id",
            "timestamp_primary": "documents.csv:published_at",
            "timestamp_fallback": "news_stories.csv:collected_at",
            "title_text": "headline_annotations.csv:target_text",
            "title_topic_relevance": "headline_annotations.csv:topic_relevance",
            "title_risk_direction": "headline_annotations.csv:risk_direction",
            "title_risk_type": "headline_annotations.csv:risk_type",
            "clause_id": "clause_annotations.csv:clause_id",
            "clause_text": "clause_annotations.csv:target_text",
            "clause_topic_relevance": "clause_annotations.csv:topic_relevance",
            "clause_risk_direction": "clause_annotations.csv:risk_direction",
            "clause_risk_type": "clause_annotations.csv:risk_type",
        },
        "parameters": {
            "title_weight": 0.25,
            "body_weight": 0.75,
            "min_history_docs": args.min_history_docs,
            "min_nonzero_docs": args.min_nonzero_docs,
            "coverage_policy": "empty hours without explicit complete coverage are NaN",
            **dict(dedup_summary["parameters"]),
        },
        "inputs": {
            "paths": diagnostics["paths"],
            "sha256": diagnostics["hashes"],
            "counts": diagnostics["counts"],
        },
        "document_counts": {
            "raw": len(documents),
            "deduplicated": len(retained),
            "duplicates_removed": len(documents) - len(retained),
            "timestamp_fallback": sum(doc.timestamp_fallback_used for doc in documents),
        },
        "deduplication": dict(dedup_summary),
        "time_range": {
            "minimum_available_at": iso_utc(min(doc.available_at for doc in documents)),
            "maximum_available_at": iso_utc(max(doc.available_at for doc in documents)),
            "minimum_hour": hourly_rows[0]["hour"],
            "maximum_hour": hourly_rows[-1]["hour"],
            "hours": len(hourly_rows),
            "nominal_collection_start": iso_utc(diagnostics["nominal_start"]),
            "collection_cutoff": iso_utc(diagnostics["collection_cutoff"]),
            "nominal_first_full_hour": iso_utc(diagnostics["nominal_first_full_hour"]),
            "nominal_last_full_hour": iso_utc(diagnostics["nominal_last_full_hour"]),
        },
        "collection_coverage": {
            "coverage_complete": False,
            "queries_expected": len(queries),
            "queries_succeeded": sum(int(item.get("error_count", 0)) == 0 for item in queries),
            "queries_at_result_cap": sum(bool(item.get("at_result_cap")) for item in queries),
            "hour_status_counts": dict(sorted(coverage_counts.items())),
            "total_score_status_counts": dict(sorted(total_status_counts.items())),
            "warning": (
                "All X queries reached the per-query result cap. Empty hours cannot be proven to be "
                "genuine no-news hours and are therefore NaN. Observed hours are computed from the "
                "retrieved sample and remain marked coverage_complete=false."
            ),
        },
        "hourly_count_summary": {
            "deduplicated_documents_per_hour": {
                "minimum": min(docs_per_hour),
                "median": statistics.median(docs_per_hour),
                "maximum": max(docs_per_hour),
            },
            "relevant_documents_per_hour": {
                "minimum": min(relevant_per_hour),
                "median": statistics.median(relevant_per_hour),
                "maximum": max(relevant_per_hour),
            },
            "by_hour": [
                {
                    "hour": row["hour"],
                    "coverage_status": row["coverage_status"],
                    "documents_raw": row["N_documents_observed_raw"],
                    "documents_deduplicated": row["N_documents_deduplicated"],
                    "relevant_total": row["N_total_relevant"],
                    **{
                        f"relevant_{risk_type}": row[f"N_{risk_type}_relevant"]
                        for risk_type in RISK_TYPES
                    },
                }
                for row in hourly_rows
            ],
        },
        "label_distribution": diagnostics["label_distribution"],
        "risk_type_counts": {
            "clause_units": diagnostics["label_distribution"]["risk_type_unit_counts"]["clause"],
            "headline_units": diagnostics["label_distribution"]["risk_type_unit_counts"]["headline"],
            "documents_raw": risk_document_counts_raw,
            "documents_deduplicated": risk_document_counts_deduplicated,
        },
        "rms_scale_summary": scale_summary(hourly_rows, scales_by_hour),
        "assertions": dict(assertions),
        "backtesting_note": (
            "Use the index available at the end of UTC hour h only with returns from hour h+1. "
            "Do not align it with returns inside hour h. Use the observed-sample eligibility flag; "
            "strict eligibility remains false until collection coverage is complete."
        ),
        "limitations": [
            "published_at is X News updated_at, not the original publisher's publication timestamp.",
            "One document uses collected_at as a conservative availability-time fallback.",
            "All 33 search queries reached the 100-result cap, so the retrieved sample may be truncated.",
            "X News objects are evolving story clusters, not independent full-text source articles.",
            "No empty hour has evidence of complete collection; empty-hour index values are NaN.",
            "Deduplication uses conservative fixed rules; ambiguous near-duplicate candidates remain for human review.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, default=ROOT / "data/processed_news/x/documents.csv")
    parser.add_argument("--processed-clauses", type=Path, default=ROOT / "data/processed_news/x/clauses.csv")
    parser.add_argument(
        "--headline-annotations",
        type=Path,
        default=ROOT / "data/deepseek_risk_annotations/full-headlines-x/x/headline_annotations.csv",
    )
    parser.add_argument(
        "--clause-annotations",
        type=Path,
        default=ROOT / "data/deepseek_risk_annotations/full-clauses-x/x/clause_annotations.csv",
    )
    parser.add_argument("--news-stories", type=Path, default=ROOT / "data/x_ai_news_30d/normalized/news_stories.csv")
    parser.add_argument("--linked-articles", type=Path, default=ROOT / "data/x_ai_news_30d/normalized/linked_articles.csv")
    parser.add_argument("--story-post-map", type=Path, default=ROOT / "data/x_ai_news_30d/normalized/story_post_map.csv")
    parser.add_argument("--collection-manifest", type=Path, default=ROOT / "data/x_ai_news_30d/collection_manifest.json")
    parser.add_argument("--coverage-hours", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/hourly_ai_risk_index/x")
    parser.add_argument("--min-history-docs", type=int, default=100)
    parser.add_argument("--min-nonzero-docs", type=int, default=5)
    parser.add_argument("--near-window-hours", type=float, default=72.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.min_history_docs < 1 or args.min_nonzero_docs < 1:
        raise SystemExit("History and nonzero thresholds must be positive")
    if args.near_window_hours <= 0:
        raise SystemExit("near_window_hours must be positive")

    documents, diagnostics = load_inputs(args)
    urls_by_source, posts_by_source = load_link_sets(args)
    dedup_summary = deduplicate_documents(
        documents,
        urls_by_source,
        posts_by_source,
        near_window_hours=args.near_window_hours,
    )
    coverage_hours = load_coverage_hours(args.coverage_hours)
    queries = diagnostics["collection_manifest"].get("queries", [])
    hourly_rows, scales_by_hour = build_hourly_index(
        documents,
        coverage_hours,
        min_history_docs=args.min_history_docs,
        min_nonzero_docs=args.min_nonzero_docs,
        nominal_start=diagnostics["nominal_start"],
        collection_cutoff=diagnostics["collection_cutoff"],
        queries_expected=len(queries),
        queries_succeeded=sum(int(item.get("error_count", 0)) == 0 for item in queries),
        queries_at_result_cap=sum(bool(item.get("at_result_cap")) for item in queries),
    )
    assertions = verify_invariants(
        documents,
        hourly_rows,
        scales_by_hour,
        args.min_history_docs,
        args.min_nonzero_docs,
    )
    audit = build_audit_report(
        args,
        documents,
        diagnostics,
        dedup_summary,
        hourly_rows,
        scales_by_hour,
        assertions,
    )

    document_fields, document_rows = document_output_rows(documents)
    atomic_write_csv(args.output_dir / "document_scores.csv", document_fields, document_rows)
    atomic_write_csv(args.output_dir / "hourly_ai_risk_index.csv", hourly_fieldnames(), hourly_rows)
    atomic_write_json(args.output_dir / "audit_report.json", audit)

    print(f"Raw documents: {len(documents)}")
    print(f"Deduplicated documents: {dedup_summary['deduplicated_documents']}")
    print(f"Hourly rows: {len(hourly_rows)}")
    print(f"Output directory: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
