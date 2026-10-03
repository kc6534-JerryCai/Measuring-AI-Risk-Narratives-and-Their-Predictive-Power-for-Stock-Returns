"""One-call composition of article preprocessing and FinBERT input creation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from spacy.language import Language
from transformers import PreTrainedTokenizerBase

from .clause_splitting import split_article_clauses
from .finbert_tokenization import tokenize_records
from .html_extraction import extract_article_text
from .record_building import build_text_records
from .sentence_splitting import split_article_sentences
from .tensor_building import build_model_tensors
from .types import ArticleText, FinBertInputConfig, ModelReadyInputs
from .windowing import build_token_windows


@dataclass(frozen=True)
class PreparedArticleInputs:
    """Traceable text outputs plus tensors ready for, but not sent to, FinBERT."""

    article: ArticleText
    finbert: ModelReadyInputs


def prepare_article_for_finbert(
    html_content: str,
    title: str,
    document_id: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    nlp: Language | None = None,
    config: FinBertInputConfig = FinBertInputConfig(),
    include_title: bool = True,
    include_clauses: bool = True,
    device: torch.device | str | None = None,
) -> PreparedArticleInputs:
    """Run the complete project flow up to the FinBERT call boundary."""

    cleaned = extract_article_text(html_content, title, document_id)
    sentences = split_article_sentences(cleaned, nlp=nlp)
    article = split_article_clauses(sentences)
    records = build_text_records(
        article,
        include_title=include_title,
        include_clauses=include_clauses,
    )
    tokenized = tokenize_records(records, tokenizer)
    windowed = build_token_windows(tokenized, tokenizer, config=config)
    finbert = build_model_tensors(
        windowed,
        tokenizer,
        device=device,
    )
    return PreparedArticleInputs(article=article, finbert=finbert)
