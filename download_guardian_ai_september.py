"""Download September 2026 Guardian AI articles using the project's client."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import types
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
MODULE_PATH = PROJECT_ROOT / "moduals" / "guardian_news_search.py"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "guardian_ai_2026_09"

KEYWORDS = (
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


def load_guardian_module() -> types.ModuleType:
    name = "guardian_news_search_standalone"
    module = types.ModuleType(name)
    module.__file__ = str(MODULE_PATH)
    sys.modules[name] = module
    code = compile(MODULE_PATH.read_bytes(), str(MODULE_PATH), "exec")
    exec(code, module.__dict__)
    return module


guardian = load_guardian_module()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def guardian_query(keyword: str) -> str:
    return f'"{keyword}"' if " " in keyword else keyword


def slug(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return (text or "query")[:50]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def raw_path(raw_dir: Path, index: int, keyword: str) -> Path:
    digest = hashlib.sha256(keyword.encode("utf-8")).hexdigest()[:8]
    return raw_dir / f"{index:02d}_{slug(keyword)}_{digest}.json"


def append_error(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--from-date", default="2026-09-01")
    parser.add_argument("--to-date", default="2026-09-30")
    parser.add_argument("--max-results", type=int, default=100)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.max_results <= 100:
        raise SystemExit("--max-results must be between 1 and 100")

    guardian.load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("GUARDIAN_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing GUARDIAN_API_KEY in the environment or .env")

    output_dir = args.output_dir.resolve()
    raw_dir = output_dir / "raw" / "search_responses"
    normalized_dir = output_dir / "normalized"
    error_path = output_dir / "collection_errors.jsonl"
    collected_at = utc_now()
    client = guardian.GuardianClient(api_key)

    articles_by_id: dict[str, Any] = {}
    matched_queries: dict[str, set[str]] = defaultdict(set)
    query_map: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    completed_queries = 0

    for index, keyword in enumerate(KEYWORDS, start=1):
        query = guardian_query(keyword)
        config = guardian.GuardianSearchConfig(
            query=query,
            from_date=args.from_date,
            to_date=args.to_date,
            language="en",
            order_by="newest",
            page_size=args.max_results,
            max_pages=1,
            max_results=args.max_results,
        )
        path = raw_path(raw_dir, index, keyword)
        reused = path.exists() and not args.refresh
        try:
            if reused:
                payload = read_json(path)
            else:
                if args.offline:
                    raise FileNotFoundError(f"Missing raw response: {path}")
                payload = client.get_search_page(
                    guardian.build_search_parameters(config, page=1)
                )
                atomic_json(path, payload)
            response = guardian._response_object(payload)
            results = response.get("results")
            if not isinstance(results, list):
                raise RuntimeError("Guardian response results were not a list")

            count = 0
            for rank, raw_result in enumerate(results, start=1):
                if not isinstance(raw_result, Mapping):
                    continue
                article = guardian.article_from_result(raw_result)
                if not article.guardian_id:
                    continue
                count += 1
                articles_by_id.setdefault(article.guardian_id, article)
                matched_queries[article.guardian_id].add(keyword)
                query_map.append(
                    {
                        "keyword": keyword,
                        "guardian_query": query,
                        "guardian_id": article.guardian_id,
                        "result_rank": rank,
                    }
                )

            total = int(response.get("total") or 0)
            query_rows.append(
                {
                    "keyword": keyword,
                    "guardian_query": query,
                    "result_count": count,
                    "guardian_total": total,
                    "at_result_cap": count >= args.max_results and total > count,
                    "raw_file": path.relative_to(PROJECT_ROOT).as_posix(),
                    "reused": reused,
                    "status": "success",
                }
            )
            completed_queries += 1
            print(f"[{index}/{len(KEYWORDS)}] {keyword}: {count} of {total}")
        except Exception as exc:
            query_rows.append(
                {
                    "keyword": keyword,
                    "guardian_query": query,
                    "result_count": 0,
                    "guardian_total": "",
                    "at_result_cap": "",
                    "raw_file": path.relative_to(PROJECT_ROOT).as_posix(),
                    "reused": reused,
                    "status": "error",
                }
            )
            append_error(
                error_path,
                {
                    "occurred_at": utc_now(),
                    "keyword": keyword,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
            )
            print(f"[{index}/{len(KEYWORDS)}] {keyword}: ERROR {exc}")

    articles = [articles_by_id[key] for key in sorted(articles_by_id)]
    saved = guardian.save_guardian_articles(
        articles,
        output_dir,
        save_body_html=True,
        overwrite=True,
    )

    generated_metadata_path = saved.metadata_csv
    with generated_metadata_path.open("r", encoding="utf-8-sig", newline="") as stream:
        metadata_rows = list(csv.DictReader(stream))
    for row in metadata_rows:
        article_id = row["guardian_id"]
        row["matched_queries"] = "|".join(sorted(matched_queries[article_id]))
        row["matched_query_count"] = len(matched_queries[article_id])
        row["collected_at"] = collected_at
    metadata_fields = list(metadata_rows[0].keys()) if metadata_rows else [
        "guardian_id", "matched_queries", "matched_query_count", "collected_at"
    ]
    metadata_path = normalized_dir / "guardian_articles.csv"
    write_csv(metadata_path, metadata_rows, metadata_fields)
    generated_metadata_path.unlink()
    write_csv(
        normalized_dir / "query_article_map.csv",
        query_map,
        ("keyword", "guardian_query", "guardian_id", "result_rank"),
    )
    write_csv(
        normalized_dir / "queries.csv",
        query_rows,
        (
            "keyword", "guardian_query", "result_count", "guardian_total",
            "at_result_cap", "raw_file", "reused", "status",
        ),
    )

    body_success = sum(1 for article in articles if article.body_html.strip())
    dates = [article.web_publication_date for article in articles if article.web_publication_date]
    manifest = {
        "collection_started_or_resumed_at": collected_at,
        "collection_finished_at": utc_now(),
        "source": "Guardian Content API",
        "endpoint": "/search",
        "from_date": args.from_date,
        "to_date": args.to_date,
        "language": "en",
        "order_by": "newest",
        "max_results_per_query": args.max_results,
        "query_count": len(KEYWORDS),
        "completed_queries": completed_queries,
        "failed_queries": len(KEYWORDS) - completed_queries,
        "raw_result_rows": len(query_map),
        "unique_articles": len(articles),
        "articles_with_body_html": body_success,
        "articles_missing_body_html": len(articles) - body_success,
        "earliest_publication": min(dates) if dates else None,
        "latest_publication": max(dates) if dates else None,
        "queries_at_result_cap": sum(
            1 for row in query_rows if row["at_result_cap"] is True
        ),
        "request_count_this_run": client.request_count,
        "metadata_csv": str(metadata_path),
        "body_directory": str(saved.body_directory),
    }
    atomic_json(output_dir / "collection_manifest.json", manifest)
    print(
        f"Saved {len(articles)} unique articles ({body_success} with body HTML) "
        f"to {output_dir}"
    )
    return 0 if completed_queries == len(KEYWORDS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
