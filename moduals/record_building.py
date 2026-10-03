"""Step 4: create independent title and clause records for FinBERT."""

from __future__ import annotations

from .types import ArticleText, FinBertTextRecord


def build_text_records(
    article: ArticleText,
    *,
    include_title: bool = True,
    include_clauses: bool = True,
) -> tuple[FinBertTextRecord, ...]:
    """Build records without concatenating the title with any clause."""

    records: list[FinBertTextRecord] = []
    if include_title:
        if not article.title:
            raise ValueError("A title record was requested, but title is empty.")
        records.append(
            FinBertTextRecord(
                record_id=f"{article.document_id}::title",
                document_id=article.document_id,
                kind="title",
                text=article.title,
            )
        )
    if include_clauses:
        records.extend(
            FinBertTextRecord(
                record_id=clause.clause_id,
                document_id=article.document_id,
                kind="clause",
                text=clause.clause_text,
            )
            for clause in article.clauses
        )
    if not records:
        raise ValueError("At least one of include_title/include_clauses must be true.")
    return tuple(records)
