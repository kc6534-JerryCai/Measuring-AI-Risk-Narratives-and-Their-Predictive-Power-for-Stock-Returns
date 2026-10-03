"""Download the last 30 days of AI news clusters from the official X API.

The script deliberately keeps raw API payloads before producing normalized CSV
tables. Run it through the project-root launcher so the existing ``types.py``
module does not shadow Python's standard-library module:

    python download_x_ai_news.py

Authentication is read from ``X_BEARER_TOKEN`` in the process environment or
from a project-root ``.env`` file.  The token is never written to output files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen


API_ROOT = "https://api.x.com/2"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "x_ai_news_30d"

DEFAULT_QUERIES = (
    "artificial intelligence",
    "generative AI",
    "large language model",
    "AI agent",
    "machine learning",
    "OpenAI",
    "ChatGPT",
    "Anthropic",
    "Claude AI",
    "Google Gemini",
    "Google DeepMind",
    "Meta AI",
    "Microsoft Copilot",
    "xAI Grok",
    "NVIDIA AI",
    "AI safety",
    "AI regulation",
    "AI copyright",
    "AI deepfake",
    "AI misinformation",
    "AI privacy",
    "AI cybersecurity",
    "AI bias",
    "AI jobs",
    "AI model release",
    "AI research",
    "AI chip",
    "AI funding",
    "AI acquisition",
    "AI partnership",
    "AI data center",
    "AI benchmark",
    "AI adoption",
)

NEWS_FIELDS = (
    "id",
    "name",
    "summary",
    "hook",
    "category",
    "keywords",
    "contexts",
    "cluster_posts_results",
    "disclaimer",
    "updated_at",
)

POST_FIELDS = (
    "id",
    "text",
    "author_id",
    "conversation_id",
    "created_at",
    "entities",
    "lang",
    "possibly_sensitive",
    "public_metrics",
    "referenced_tweets",
)

USER_FIELDS = (
    "id",
    "name",
    "username",
    "description",
    "location",
    "verified",
    "verified_type",
    "public_metrics",
    "created_at",
    "protected",
    "url",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_dotenv(path: Path) -> None:
    """Load simple KEY=VALUE entries without printing secret values."""

    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def slug(value: str, *, limit: int = 45) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return (normalized or "query")[:limit]


def chunks(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


class XApiClient:
    """Small rate-aware X API client that never puts the token in the URL."""

    def __init__(
        self,
        bearer_token: str,
        *,
        timeout_seconds: float = 60.0,
        max_retries: int = 5,
    ) -> None:
        if not bearer_token.strip():
            raise ValueError("X_BEARER_TOKEN is empty.")
        self.bearer_token = bearer_token.strip()
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.request_count = 0

    def get_json(self, endpoint: str, parameters: Mapping[str, Any]) -> Mapping[str, Any]:
        url = f"{API_ROOT}{endpoint}?{urlencode(parameters)}"
        for attempt in range(1, self.max_retries + 1):
            request = Request(
                url,
                headers={
                    "Authorization": f"Bearer {self.bearer_token}",
                    "User-Agent": "NLP-assignment4-x-ai-news/1.0",
                },
            )
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    self.request_count += 1
                    payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, Mapping):
                    raise RuntimeError("X API returned a non-object JSON response.")
                return payload
            except HTTPError as exc:
                self.request_count += 1
                if exc.code in {401, 403}:
                    raise RuntimeError(
                        f"X API authentication/access failed with HTTP {exc.code}."
                    ) from exc
                if exc.code == 429:
                    retry_after = exc.headers.get("Retry-After", "60")
                    try:
                        delay = max(1.0, float(retry_after))
                    except ValueError:
                        delay = 60.0
                elif 500 <= exc.code < 600:
                    delay = min(60.0, 2.0**attempt)
                else:
                    detail = exc.read().decode("utf-8", errors="replace")[:500]
                    raise RuntimeError(
                        f"X API returned HTTP {exc.code}: {detail}"
                    ) from exc
            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
                self.request_count += 1
                delay = min(60.0, 2.0**attempt)
                if attempt == self.max_retries:
                    raise RuntimeError("X API request repeatedly failed.") from exc
            if attempt == self.max_retries:
                raise RuntimeError("X API request repeatedly failed.")
            time.sleep(delay)
        raise AssertionError("Unreachable retry state")


def news_query_path(raw_dir: Path, index: int, query: str) -> Path:
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:8]
    return raw_dir / f"{index:02d}_{slug(query)}_{digest}.json"


def download_news_queries(
    client: XApiClient,
    queries: Sequence[str],
    raw_dir: Path,
    *,
    max_age_hours: int,
    max_results: int,
    refresh: bool,
    offline: bool,
) -> tuple[list[tuple[str, Mapping[str, Any]]], list[dict[str, Any]]]:
    payloads: list[tuple[str, Mapping[str, Any]]] = []
    audit: list[dict[str, Any]] = []
    for index, query in enumerate(queries, start=1):
        path = news_query_path(raw_dir, index, query)
        reused = path.exists() and not refresh
        if reused:
            payload = read_json(path)
        else:
            if offline:
                raise FileNotFoundError(
                    f"Offline normalization is missing raw news response: {path}"
                )
            payload = client.get_json(
                "/news/search",
                {
                    "query": query,
                    "max_results": max_results,
                    "max_age_hours": max_age_hours,
                    "news.fields": ",".join(NEWS_FIELDS),
                },
            )
            atomic_json(path, payload)
        data = payload.get("data")
        result_count = len(data) if isinstance(data, list) else 0
        audit.append(
            {
                "query": query,
                "raw_file": path.relative_to(PROJECT_ROOT).as_posix(),
                "result_count": result_count,
                "at_result_cap": result_count >= max_results,
                "reused": reused,
                "error_count": len(payload.get("errors", []))
                if isinstance(payload.get("errors"), list)
                else 0,
            }
        )
        payloads.append((query, payload))
        print(f"[{index}/{len(queries)}] {query}: {result_count} news stories")
    return payloads, audit


def normalize_news(
    payloads: Sequence[tuple[str, Mapping[str, Any]]],
    collected_at: str,
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    merged: dict[str, dict[str, Any]] = {}
    story_posts: dict[str, set[str]] = defaultdict(set)
    for query, payload in payloads:
        data = payload.get("data")
        if not isinstance(data, list):
            continue
        for item in data:
            if not isinstance(item, Mapping):
                continue
            news_id = str(item.get("id") or "").strip()
            if not news_id:
                continue
            row = merged.setdefault(
                news_id,
                {
                    "news_id": news_id,
                    "headline": str(item.get("name") or ""),
                    "summary": str(item.get("summary") or ""),
                    "hook": str(item.get("hook") or ""),
                    "category": str(item.get("category") or ""),
                    "keywords": item.get("keywords") or {},
                    "contexts": item.get("contexts") or {},
                    "disclaimer": str(item.get("disclaimer") or ""),
                    "updated_at": str(item.get("updated_at") or ""),
                    "collected_at": collected_at,
                    "matched_queries": set(),
                },
            )
            row["matched_queries"].add(query)
            cluster = item.get("cluster_posts_results")
            if isinstance(cluster, list):
                for link in cluster:
                    if isinstance(link, Mapping):
                        post_id = str(link.get("post_id") or "").strip()
                        if post_id:
                            story_posts[news_id].add(post_id)

    rows: list[dict[str, Any]] = []
    for news_id in sorted(merged):
        row = merged[news_id]
        rows.append(
            {
                **row,
                "keywords": json_cell(row["keywords"]),
                "contexts": json_cell(row["contexts"]),
                "matched_queries": "|".join(sorted(row["matched_queries"])),
                "cluster_post_count": len(story_posts[news_id]),
                "event_time": str(row["updated_at"]),
                "event_time_basis": "x_news_updated_at",
                "earliest_recent_cluster_post_at": "",
            }
        )
    return rows, story_posts


def prioritize_story_post_ids(
    news_rows: Sequence[Mapping[str, Any]],
    story_posts: Mapping[str, set[str]],
) -> list[str]:
    """Round-robin IDs so partial downloads cover many stories, newest first."""

    ordered_stories = sorted(
        news_rows,
        key=lambda row: str(row.get("updated_at") or ""),
        reverse=True,
    )
    queues = [sorted(story_posts.get(str(row["news_id"]), set())) for row in ordered_stories]
    prioritized: list[str] = []
    seen: set[str] = set()
    offset = 0
    while True:
        added = False
        for queue in queues:
            if offset >= len(queue):
                continue
            added = True
            post_id = queue[offset]
            if post_id not in seen:
                seen.add(post_id)
                prioritized.append(post_id)
        if not added:
            break
        offset += 1
    return prioritized


def download_post_batches(
    client: XApiClient,
    post_ids: Sequence[str],
    raw_dir: Path,
    *,
    refresh: bool,
    offline: bool,
) -> tuple[list[Mapping[str, Any]], str | None]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    existing_paths = sorted(raw_dir.glob("batch_*.json"))
    payloads: list[Mapping[str, Any]] = [read_json(path) for path in existing_paths]
    downloaded_ids: set[str] = set()
    for payload in payloads:
        data = payload.get("data")
        if isinstance(data, list):
            downloaded_ids.update(
                str(item.get("id"))
                for item in data
                if isinstance(item, Mapping) and item.get("id")
            )
    remaining_ids = [post_id for post_id in post_ids if post_id not in downloaded_ids]
    stopped_reason: str | None = None
    batches = list(chunks(remaining_ids, 100))
    starting_index = len(existing_paths) + 1
    if offline and batches:
        stopped_reason = (
            f"offline mode: {len(remaining_ids)} requested post IDs have no saved object"
        )
        print(f"[post lookup] stopped: {stopped_reason}")
        return payloads, stopped_reason
    for offset, batch in enumerate(batches):
        index = starting_index + offset
        digest = hashlib.sha256(",".join(batch).encode("utf-8")).hexdigest()[:10]
        path = raw_dir / f"batch_{index:04d}_{digest}.json"
        try:
            payload = client.get_json(
                "/tweets",
                {
                    "ids": ",".join(batch),
                    "tweet.fields": ",".join(POST_FIELDS),
                    "expansions": "author_id",
                    "user.fields": ",".join(USER_FIELDS),
                },
            )
        except RuntimeError as exc:
            stopped_reason = str(exc)
            print(f"[post batch {offset + 1}/{len(batches)}] stopped: {stopped_reason}")
            break
        atomic_json(path, payload)
        payloads.append(payload)
        data = payload.get("data")
        count = len(data) if isinstance(data, list) else 0
        print(f"[post batch {offset + 1}/{len(batches)}] {count}/{len(batch)} posts")
    return payloads, stopped_reason


def canonicalize_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return value
    kept = []
    for pair in parsed.query.split("&") if parsed.query else []:
        name = pair.split("=", 1)[0].lower()
        if name.startswith("utm_") or name in {"ref", "source", "campaign"}:
            continue
        kept.append(pair)
    return urlunsplit(
        (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, "&".join(kept), "")
    )


def normalize_posts_and_authors(
    payloads: Sequence[Mapping[str, Any]],
    story_posts: Mapping[str, set[str]],
    collected_at: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    post_to_stories: dict[str, set[str]] = defaultdict(set)
    for news_id, post_ids in story_posts.items():
        for post_id in post_ids:
            post_to_stories[post_id].add(news_id)

    posts: dict[str, dict[str, Any]] = {}
    authors: dict[str, dict[str, Any]] = {}
    articles: dict[tuple[str, str, str], dict[str, Any]] = {}

    for payload in payloads:
        data = payload.get("data")
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, Mapping):
                    continue
                post_id = str(item.get("id") or "").strip()
                if not post_id:
                    continue
                metrics = item.get("public_metrics")
                if not isinstance(metrics, Mapping):
                    metrics = {}
                entities = item.get("entities")
                if not isinstance(entities, Mapping):
                    entities = {}
                urls = entities.get("urls")
                if not isinstance(urls, list):
                    urls = []
                expanded_urls: list[str] = []
                for url_item in urls:
                    if not isinstance(url_item, Mapping):
                        continue
                    expanded = str(
                        url_item.get("unwound_url")
                        or url_item.get("expanded_url")
                        or url_item.get("url")
                        or ""
                    ).strip()
                    if not expanded:
                        continue
                    expanded_urls.append(expanded)
                    canonical = canonicalize_url(expanded)
                    domain = urlsplit(canonical).netloc.lower()
                    for news_id in post_to_stories.get(post_id, set()):
                        key = (news_id, post_id, canonical)
                        articles[key] = {
                            "news_id": news_id,
                            "post_id": post_id,
                            "original_url": str(url_item.get("url") or ""),
                            "expanded_url": expanded,
                            "canonical_url": canonical,
                            "source_domain": domain,
                            "page_title": str(url_item.get("title") or ""),
                            "page_description": str(url_item.get("description") or ""),
                            "retrieval_status": "metadata_only",
                            "retrieved_at": collected_at,
                        }
                references = item.get("referenced_tweets") or item.get("referenced_posts") or []
                posts[post_id] = {
                    "post_id": post_id,
                    "author_id": str(item.get("author_id") or ""),
                    "text": str(item.get("text") or ""),
                    "created_at": str(item.get("created_at") or ""),
                    "language": str(item.get("lang") or ""),
                    "conversation_id": str(item.get("conversation_id") or ""),
                    "post_url": f"https://x.com/i/web/status/{post_id}",
                    "external_urls": json_cell(expanded_urls),
                    "hashtags": json_cell(entities.get("hashtags") or []),
                    "mentions": json_cell(entities.get("mentions") or []),
                    "referenced_posts": json_cell(references),
                    "possibly_sensitive": item.get("possibly_sensitive", ""),
                    "reply_count": metrics.get("reply_count", ""),
                    "repost_count": metrics.get("retweet_count", metrics.get("repost_count", "")),
                    "like_count": metrics.get("like_count", ""),
                    "quote_count": metrics.get("quote_count", ""),
                    "bookmark_count": metrics.get("bookmark_count", ""),
                    "impression_count": metrics.get("impression_count", ""),
                    "metrics_collected_at": collected_at,
                }

        includes = payload.get("includes")
        if isinstance(includes, Mapping):
            users = includes.get("users")
            if isinstance(users, list):
                for item in users:
                    if not isinstance(item, Mapping):
                        continue
                    author_id = str(item.get("id") or "").strip()
                    if not author_id:
                        continue
                    metrics = item.get("public_metrics")
                    if not isinstance(metrics, Mapping):
                        metrics = {}
                    authors[author_id] = {
                        "author_id": author_id,
                        "username": str(item.get("username") or ""),
                        "display_name": str(item.get("name") or ""),
                        "description": str(item.get("description") or ""),
                        "location": str(item.get("location") or ""),
                        "verified": item.get("verified", ""),
                        "verified_type": str(item.get("verified_type") or ""),
                        "followers_count": metrics.get("followers_count", ""),
                        "following_count": metrics.get("following_count", ""),
                        "post_count": metrics.get("tweet_count", metrics.get("post_count", "")),
                        "account_created_at": str(item.get("created_at") or ""),
                        "protected": item.get("protected", ""),
                        "profile_url": str(item.get("url") or ""),
                        "author_collected_at": collected_at,
                    }

    mappings = [
        {"news_id": news_id, "post_id": post_id, "relationship": "cluster_evidence"}
        for news_id in sorted(story_posts)
        for post_id in sorted(story_posts[news_id])
    ]
    return (
        [posts[key] for key in sorted(posts)],
        [authors[key] for key in sorted(authors)],
        mappings,
        [articles[key] for key in sorted(articles)],
    )


def fill_recent_cluster_time(
    news_rows: list[dict[str, Any]],
    posts: Sequence[Mapping[str, Any]],
    mappings: Sequence[Mapping[str, Any]],
    *,
    collected_at: str,
    max_age_hours: int,
) -> None:
    post_times = {str(row["post_id"]): str(row.get("created_at") or "") for row in posts}
    story_times: dict[str, list[str]] = defaultdict(list)
    cutoff = datetime.fromisoformat(collected_at.replace("Z", "+00:00")) - timedelta(
        hours=max_age_hours
    )
    for mapping in mappings:
        created_at = post_times.get(str(mapping["post_id"]), "")
        if not created_at:
            continue
        try:
            parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError:
            continue
        if parsed >= cutoff:
            story_times[str(mapping["news_id"])].append(created_at)
    for row in news_rows:
        times = story_times.get(str(row["news_id"]), [])
        row["earliest_recent_cluster_post_at"] = min(times) if times else ""


NEWS_COLUMNS = (
    "news_id", "headline", "summary", "hook", "category", "keywords", "contexts",
    "disclaimer", "updated_at", "event_time", "event_time_basis",
    "earliest_recent_cluster_post_at", "cluster_post_count",
    "matched_queries", "collected_at",
)
POST_COLUMNS = (
    "post_id", "author_id", "text", "created_at", "language", "conversation_id",
    "post_url", "external_urls", "hashtags", "mentions", "referenced_posts",
    "possibly_sensitive", "reply_count", "repost_count", "like_count", "quote_count",
    "bookmark_count", "impression_count", "metrics_collected_at",
)
AUTHOR_COLUMNS = (
    "author_id", "username", "display_name", "description", "location", "verified",
    "verified_type", "followers_count", "following_count", "post_count",
    "account_created_at", "protected", "profile_url", "author_collected_at",
)
MAPPING_COLUMNS = ("news_id", "post_id", "relationship")
ARTICLE_COLUMNS = (
    "news_id", "post_id", "original_url", "expanded_url", "canonical_url",
    "source_domain", "page_title", "page_description", "retrieval_status", "retrieved_at",
)
REACTION_COLUMNS = (
    "reaction_post_id", "news_id", "target_post_id", "author_id", "text",
    "created_at", "reaction_type", "public_metrics",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--query", action="append", dest="queries", help="Override the default query list; repeat for multiple queries")
    parser.add_argument("--max-age-hours", type=int, default=720)
    parser.add_argument("--max-results", type=int, default=100)
    parser.add_argument("--refresh", action="store_true", help="Redownload existing raw response files")
    parser.add_argument("--skip-posts", action="store_true", help="Only download and normalize News API objects")
    parser.add_argument("--offline", action="store_true", help="Normalize existing raw files without making API requests")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.max_age_hours <= 720:
        raise SystemExit("--max-age-hours must be between 1 and 720")
    if not 1 <= args.max_results <= 100:
        raise SystemExit("--max-results must be between 1 and 100")

    load_dotenv(PROJECT_ROOT / ".env")
    token = os.environ.get("X_BEARER_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "Missing X_BEARER_TOKEN. Add it to the project-root .env file or the process environment."
        )

    queries = tuple(args.queries or DEFAULT_QUERIES)
    output_dir = args.output_dir.resolve()
    raw_news_dir = output_dir / "raw" / "news_search"
    raw_posts_dir = output_dir / "raw" / "post_lookup"
    normalized_dir = output_dir / "normalized"
    collected_at = utc_now()
    client = XApiClient(token)

    payloads, query_audit = download_news_queries(
        client,
        queries,
        raw_news_dir,
        max_age_hours=args.max_age_hours,
        max_results=args.max_results,
        refresh=args.refresh,
        offline=args.offline,
    )
    news_rows, story_posts = normalize_news(payloads, collected_at)

    post_rows: list[dict[str, Any]] = []
    author_rows: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    article_rows: list[dict[str, Any]] = []
    post_download_error: str | None = None
    expected_unique_post_ids = 0
    if not args.skip_posts:
        post_ids = prioritize_story_post_ids(news_rows, story_posts)
        expected_unique_post_ids = len(post_ids)
        post_payloads, post_download_error = download_post_batches(
            client,
            post_ids,
            raw_posts_dir,
            refresh=args.refresh,
            offline=args.offline,
        ) if post_ids else ([], None)
        post_rows, author_rows, mappings, article_rows = normalize_posts_and_authors(
            post_payloads,
            story_posts,
            collected_at,
        )
        fill_recent_cluster_time(
            news_rows,
            post_rows,
            mappings,
            collected_at=collected_at,
            max_age_hours=args.max_age_hours,
        )

    write_csv(normalized_dir / "news_stories.csv", news_rows, NEWS_COLUMNS)
    write_csv(normalized_dir / "evidence_posts.csv", post_rows, POST_COLUMNS)
    write_csv(normalized_dir / "authors.csv", author_rows, AUTHOR_COLUMNS)
    write_csv(normalized_dir / "story_post_map.csv", mappings, MAPPING_COLUMNS)
    write_csv(normalized_dir / "linked_articles.csv", article_rows, ARTICLE_COLUMNS)
    write_csv(normalized_dir / "reaction_posts.csv", [], REACTION_COLUMNS)

    manifest = {
        "collection_started_or_resumed_at": collected_at,
        "collection_finished_at": utc_now(),
        "endpoint": "/2/news/search",
        "max_age_hours": args.max_age_hours,
        "max_results_per_query": args.max_results,
        "queries": query_audit,
        "counts": {
            "unique_news_stories": len(news_rows),
            "cluster_post_ids": sum(len(ids) for ids in story_posts.values()),
            "downloaded_evidence_posts": len(post_rows),
            "authors": len(author_rows),
            "story_post_links": len(mappings),
            "linked_article_rows": len(article_rows),
            "reaction_posts": 0,
        },
        "post_lookup_coverage": {
            "status": "partial" if post_download_error else "complete",
            "expected_unique_post_ids": expected_unique_post_ids,
            "downloaded_post_objects": len(post_rows),
            "missing_or_unavailable_post_objects": max(0, expected_unique_post_ids - len(post_rows)),
            "stopped_reason": post_download_error,
        },
        "reaction_coverage": {
            "status": "not_collected",
            "reason": "Thirty-day reactions require a separate entitlement-aware search pass.",
        },
        "request_count_this_run": client.request_count,
        "notes": [
            "News objects are X-generated story clusters, not independent source articles.",
            "event_time currently uses X News updated_at because the API does not expose a publication timestamp.",
            "earliest_recent_cluster_post_at excludes cluster posts older than the requested window.",
            "A query returning max_results may be truncated and should be subdivided.",
        ],
    }
    atomic_json(output_dir / "collection_manifest.json", manifest)
    print(
        f"Saved {len(news_rows)} news stories, {len(post_rows)} evidence posts, "
        f"and {len(author_rows)} authors to {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
