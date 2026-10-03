"""Search and optionally save arbitrary Guardian news from the Content API.

This module is deliberately topic-agnostic.  The caller supplies the query,
date range, section, tag, ordering, and result limits.  It can be imported as a
library or run with ``python -m moduals.guardian_news_search``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    from bs4 import BeautifulSoup
except ModuleNotFoundError:  # Keep the downloader usable with the standard library.
    BeautifulSoup = None  # type: ignore[assignment,misc]

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv(path: Path) -> None:
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


SEARCH_ENDPOINT = "https://content.guardianapis.com/search"
DEFAULT_SHOW_FIELDS = ("trailText", "body", "byline", "wordcount")
DEFAULT_SHOW_TAGS = ("keyword",)
VALID_ORDERINGS = {"newest", "oldest", "relevance"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())


def _stdlib_html_text(value: str) -> str:
    parser = _TextExtractor()
    parser.feed(value)
    parser.close()
    return " ".join(parser.parts)


@dataclass(frozen=True)
class GuardianSearchConfig:
    """Filters and safety limits for one Guardian search."""

    query: str
    from_date: str | None = None
    to_date: str | None = None
    section: str | None = None
    tag: str | None = None
    language: str | None = "en"
    order_by: str = "newest"
    query_fields: str | None = None
    page_size: int = 200
    max_pages: int | None = None
    max_results: int | None = None
    show_fields: tuple[str, ...] = DEFAULT_SHOW_FIELDS
    show_tags: tuple[str, ...] = DEFAULT_SHOW_TAGS
    additional_parameters: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query must not be empty.")
        if not 1 <= self.page_size <= 200:
            raise ValueError("page_size must be between 1 and 200.")
        if self.order_by not in VALID_ORDERINGS:
            raise ValueError(f"order_by must be one of {sorted(VALID_ORDERINGS)}.")
        if self.max_pages is not None and self.max_pages < 1:
            raise ValueError("max_pages must be positive when supplied.")
        if self.max_results is not None and self.max_results < 1:
            raise ValueError("max_results must be positive when supplied.")
        parsed_from = _parse_date(self.from_date, "from_date")
        parsed_to = _parse_date(self.to_date, "to_date")
        if parsed_from and parsed_to and parsed_from > parsed_to:
            raise ValueError("from_date must not be later than to_date.")
        protected = {"api-key", "q", "page", "page-size"}
        overlap = protected & set(self.additional_parameters)
        if overlap:
            raise ValueError(
                f"additional_parameters cannot override {sorted(overlap)}."
            )


@dataclass(frozen=True)
class GuardianArticle:
    """Normalized metadata and optional body HTML for one search result."""

    guardian_id: str
    content_type: str
    section_id: str
    section_name: str
    web_publication_date: str
    web_title: str
    web_url: str
    api_url: str
    is_live_blog: bool
    trail_text: str
    byline: str
    wordcount: str
    keyword_tags: str
    body_html: str


@dataclass(frozen=True)
class SavedGuardianSearch:
    output_dir: Path
    metadata_csv: Path
    manifest_json: Path
    body_directory: Path | None
    article_count: int


def _parse_date(value: str | None, name: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must use YYYY-MM-DD format.") from exc


def _plain_text(value: Any) -> str:
    if not value:
        return ""
    if BeautifulSoup is not None:
        return BeautifulSoup(str(value), "html.parser").get_text(" ", strip=True)
    return html.unescape(_stdlib_html_text(str(value)))


def _keyword_tags(tags: Any) -> str:
    if not isinstance(tags, list):
        return ""
    names = [
        str(tag.get("webTitle") or "").strip()
        for tag in tags
        if isinstance(tag, dict) and tag.get("type") == "keyword"
    ]
    return "|".join(dict.fromkeys(name for name in names if name))


def article_from_result(result: Mapping[str, Any]) -> GuardianArticle:
    """Normalize one raw Guardian API result."""

    fields = result.get("fields")
    if not isinstance(fields, Mapping):
        fields = {}
    content_type = str(result.get("type") or "")
    return GuardianArticle(
        guardian_id=str(result.get("id") or ""),
        content_type=content_type,
        section_id=str(result.get("sectionId") or ""),
        section_name=str(result.get("sectionName") or ""),
        web_publication_date=str(result.get("webPublicationDate") or ""),
        web_title=str(result.get("webTitle") or ""),
        web_url=str(result.get("webUrl") or ""),
        api_url=str(result.get("apiUrl") or ""),
        is_live_blog=content_type == "liveblog",
        trail_text=_plain_text(fields.get("trailText")),
        byline=_plain_text(fields.get("byline")),
        wordcount=str(fields.get("wordcount") or ""),
        keyword_tags=_keyword_tags(result.get("tags")),
        body_html=str(fields.get("body") or ""),
    )


def build_search_parameters(
    config: GuardianSearchConfig,
    *,
    page: int,
) -> dict[str, str | int]:
    """Translate a search configuration into Guardian API parameters."""

    parameters: dict[str, str | int] = {
        "q": config.query,
        "page": page,
        "page-size": config.page_size,
        "order-by": config.order_by,
    }
    optional = {
        "from-date": config.from_date,
        "to-date": config.to_date,
        "section": config.section,
        "tag": config.tag,
        "lang": config.language,
        "query-fields": config.query_fields,
        "show-fields": ",".join(config.show_fields) if config.show_fields else None,
        "show-tags": ",".join(config.show_tags) if config.show_tags else None,
    }
    parameters.update({key: value for key, value in optional.items() if value})
    parameters.update(config.additional_parameters)
    return parameters


class GuardianClient:
    """Rate-limited Guardian API client with bounded retries and requests."""

    def __init__(
        self,
        api_key: str,
        *,
        minimum_interval_seconds: float = 1.1,
        max_requests: int = 500,
        max_retries: int = 5,
        timeout_seconds: float = 60.0,
        request_json: Callable[[str, float], Mapping[str, Any]] | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("A Guardian API key is required.")
        self.api_key = api_key.strip()
        self.minimum_interval_seconds = max(0.0, minimum_interval_seconds)
        self.max_requests = max_requests
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.request_count = 0
        self.last_request_at: float | None = None
        self._request_json = request_json or self._default_request_json

    @staticmethod
    def _default_request_json(url: str, timeout: float) -> Mapping[str, Any]:
        request = Request(
            url,
            headers={"User-Agent": "NLP-assignment3-guardian-search/1.0"},
        )
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _wait(self) -> None:
        if self.last_request_at is None:
            return
        remaining = self.minimum_interval_seconds - (
            time.monotonic() - self.last_request_at
        )
        if remaining > 0:
            time.sleep(remaining)

    def get_search_page(
        self,
        parameters: Mapping[str, str | int],
    ) -> Mapping[str, Any]:
        """Fetch one search page without exposing the API key in errors."""

        url = f"{SEARCH_ENDPOINT}?{urlencode({**parameters, 'api-key': self.api_key})}"
        delay = self.minimum_interval_seconds
        for attempt in range(1, self.max_retries + 1):
            if self.request_count >= self.max_requests:
                raise RuntimeError(
                    f"Guardian request safety limit ({self.max_requests}) reached."
                )
            self._wait()
            try:
                payload = self._request_json(url, self.timeout_seconds)
                self.last_request_at = time.monotonic()
                self.request_count += 1
                return payload
            except HTTPError as exc:
                self.last_request_at = time.monotonic()
                self.request_count += 1
                if exc.code in {401, 403}:
                    raise RuntimeError("Guardian API authentication failed.") from exc
                if exc.code == 429:
                    try:
                        delay = max(
                            float(exc.headers.get("Retry-After", "1.1")),
                            self.minimum_interval_seconds,
                        )
                    except ValueError:
                        delay = self.minimum_interval_seconds
                elif 500 <= exc.code < 600:
                    delay = min(30.0, 2.0**attempt)
                else:
                    raise RuntimeError(f"Guardian API returned HTTP {exc.code}.") from exc
            except (URLError, TimeoutError, json.JSONDecodeError) as exc:
                self.last_request_at = time.monotonic()
                self.request_count += 1
                delay = min(30.0, 2.0**attempt)
                if attempt == self.max_retries:
                    raise RuntimeError("Guardian API request repeatedly failed.") from exc
            if attempt == self.max_retries:
                raise RuntimeError("Guardian API request repeatedly failed.")
            time.sleep(delay)
        raise AssertionError("Unreachable retry state.")


def _response_object(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    response = payload.get("response")
    if not isinstance(response, Mapping) or response.get("status") != "ok":
        raise RuntimeError("Guardian API response was not successful.")
    return response


def search_guardian(
    config: GuardianSearchConfig,
    *,
    api_key: str | None = None,
    client: GuardianClient | None = None,
) -> list[GuardianArticle]:
    """Search all allowed pages and return normalized, de-duplicated articles."""

    if client is None:
        resolved_key = (api_key or os.environ.get("GUARDIAN_API_KEY", "")).strip()
        client = GuardianClient(resolved_key)

    articles: list[GuardianArticle] = []
    seen_ids: set[str] = set()
    page = 1
    total_pages = 1
    while page <= total_pages:
        if config.max_pages is not None and page > config.max_pages:
            break
        response = _response_object(
            client.get_search_page(build_search_parameters(config, page=page))
        )
        total_pages = int(response.get("pages") or 0)
        results = response.get("results")
        if not isinstance(results, list):
            raise RuntimeError(f"Guardian API results were invalid on page {page}.")
        for raw_result in results:
            if not isinstance(raw_result, Mapping):
                continue
            article = article_from_result(raw_result)
            if not article.guardian_id or article.guardian_id in seen_ids:
                continue
            seen_ids.add(article.guardian_id)
            articles.append(article)
            if config.max_results is not None and len(articles) >= config.max_results:
                return articles
        page += 1
    return articles


def _body_filename(article: GuardianArticle) -> str:
    date_part = article.web_publication_date[:10].replace("-", "") or "unknown"
    title_slug = re.sub(r"[^a-z0-9]+", "-", article.web_title.lower()).strip("-")[:50]
    title_slug = title_slug or "article"
    digest = hashlib.sha256(article.guardian_id.encode("utf-8")).hexdigest()[:12]
    return f"{date_part}_{title_slug}_{digest}.html"


def _atomic_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding=encoding)
    temporary.replace(path)


def save_guardian_articles(
    articles: Sequence[GuardianArticle],
    output_dir: Path,
    *,
    config: GuardianSearchConfig | None = None,
    save_body_html: bool = True,
    overwrite: bool = False,
) -> SavedGuardianSearch:
    """Save metadata, a reproducibility manifest, and optional body HTML files."""

    output_dir = output_dir.resolve()
    metadata_path = output_dir / "guardian_articles.csv"
    manifest_path = output_dir / "search_manifest.json"
    existing = [path for path in (metadata_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Search outputs already exist; pass overwrite=True to replace them: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    body_dir = output_dir / "body_html" if save_body_html else None
    if body_dir is not None:
        body_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "guardian_id",
        "content_type",
        "section_id",
        "section_name",
        "web_publication_date",
        "web_title",
        "web_url",
        "api_url",
        "is_live_blog",
        "trail_text",
        "byline",
        "wordcount",
        "keyword_tags",
        "body_html_path",
        "download_status",
    ]
    temporary_csv = metadata_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for article in articles:
            body_path = ""
            status = "metadata_only"
            if save_body_html:
                if article.body_html.strip():
                    path = body_dir / _body_filename(article)  # type: ignore[operator]
                    _atomic_text(path, article.body_html)
                    body_path = path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else str(path)
                    status = "success"
                else:
                    status = "missing_body"
            row = asdict(article)
            row.pop("body_html")
            row.update(body_html_path=body_path, download_status=status)
            writer.writerow(row)
    temporary_csv.replace(metadata_path)

    manifest = {
        "article_count": len(articles),
        "body_html_saved": save_body_html,
        "query_config": asdict(config) if config is not None else None,
        "metadata_csv": str(metadata_path),
        "body_directory": str(body_dir) if body_dir is not None else None,
    }
    _atomic_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2))
    return SavedGuardianSearch(
        output_dir=output_dir,
        metadata_csv=metadata_path,
        manifest_json=manifest_path,
        body_directory=body_dir,
        article_count=len(articles),
    )


def search_and_save(
    config: GuardianSearchConfig,
    output_dir: Path,
    *,
    api_key: str | None = None,
    client: GuardianClient | None = None,
    save_body_html: bool = True,
    overwrite: bool = False,
) -> SavedGuardianSearch:
    articles = search_guardian(config, api_key=api_key, client=client)
    return save_guardian_articles(
        articles,
        output_dir,
        config=config,
        save_body_html=save_body_html,
        overwrite=overwrite,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Guardian search expression")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--from-date", help="YYYY-MM-DD")
    parser.add_argument("--to-date", help="YYYY-MM-DD")
    parser.add_argument("--section")
    parser.add_argument("--tag")
    parser.add_argument("--language", default="en")
    parser.add_argument("--order-by", choices=sorted(VALID_ORDERINGS), default="newest")
    parser.add_argument("--query-fields")
    parser.add_argument("--page-size", type=int, default=200)
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--max-results", type=int)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    config = GuardianSearchConfig(
        query=arguments.query,
        from_date=arguments.from_date,
        to_date=arguments.to_date,
        section=arguments.section,
        tag=arguments.tag,
        language=arguments.language,
        order_by=arguments.order_by,
        query_fields=arguments.query_fields,
        page_size=arguments.page_size,
        max_pages=arguments.max_pages,
        max_results=arguments.max_results,
    )
    saved = search_and_save(
        config,
        arguments.output_dir,
        save_body_html=not arguments.metadata_only,
        overwrite=arguments.overwrite,
    )
    print(f"Saved {saved.article_count} articles to {saved.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
