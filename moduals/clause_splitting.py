"""Step 3: conservatively split parsed sentences into clauses."""

from __future__ import annotations

from src.preprocess_financial_news import split_clauses

from .types import ArticleText, SentenceArticle


def split_article_clauses(article: SentenceArticle) -> ArticleText:
    """Apply the project's syntax-aware clause-boundary rules."""

    clauses = []
    for parsed_sentence, sentence_record in zip(
        article.parsed_sentences,
        article.sentences,
        strict=True,
    ):
        clauses.extend(
            split_clauses(
                parsed_sentence,
                sentence_record,
                article.cleaned_document_text,
            )
        )
    return ArticleText(
        document_id=article.document_id,
        title=article.title,
        cleaned_document_text=article.cleaned_document_text,
        sentences=article.sentences,
        clauses=tuple(clauses),
    )
