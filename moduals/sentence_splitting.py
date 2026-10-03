"""Step 2: split cleaned article text into traceable sentences."""

from __future__ import annotations

from spacy.language import Language

from src.preprocess_financial_news import load_nlp, split_sentences

from .types import CleanedArticle, SentenceArticle


def split_article_sentences(
    article: CleanedArticle,
    *,
    nlp: Language | None = None,
) -> SentenceArticle:
    """Run the project spaCy parser and retain document-relative offsets."""

    parser = nlp if nlp is not None else load_nlp()
    parsed_document, records = split_sentences(
        article.cleaned_document_text,
        parser,
        article.document_id,
    )
    parsed_sentences = tuple(parsed_document.sents)
    if len(parsed_sentences) != len(records):
        raise RuntimeError("spaCy sentence spans do not match sentence records.")
    return SentenceArticle(
        document_id=article.document_id,
        title=article.title,
        cleaned_document_text=article.cleaned_document_text,
        sentences=tuple(records),
        parsed_sentences=parsed_sentences,
    )
