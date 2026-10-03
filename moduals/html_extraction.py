"""Step 1: extract and minimally normalize article text from HTML."""

from __future__ import annotations

from src.preprocess_financial_news import extract_main_text

from .types import CleanedArticle


def extract_article_text(
    html_content: str,
    title: str,
    document_id: str,
) -> CleanedArticle:
    """Remove webpage noise, prepend the title, and preserve paragraphs."""

    if not document_id.strip():
        raise ValueError("document_id must not be empty.")
    return CleanedArticle(
        document_id=document_id,
        title=title.strip(),
        cleaned_document_text=extract_main_text(html_content, title),
    )
