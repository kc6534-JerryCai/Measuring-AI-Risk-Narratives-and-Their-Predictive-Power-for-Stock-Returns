"""Reusable preparation steps that stop immediately before FinBERT inference.

The directory name follows the spelling requested for this project.  Import the
public API from here instead of depending on the individual implementation
modules.
"""

from .clause_splitting import split_article_clauses
from .finbert_tokenization import load_tokenizer, tokenize_records
from .html_extraction import extract_article_text
from .pipeline import PreparedArticleInputs, prepare_article_for_finbert
from .record_building import build_text_records
from .sentence_splitting import split_article_sentences
from .tensor_building import build_model_tensors
from .types import (
    ArticleText,
    FinBertInputConfig,
    FinBertTextRecord,
    ModelReadyInputs,
)
from .windowing import build_token_windows


def prepare_model_inputs(records, tokenizer, *, config=FinBertInputConfig(), device=None):
    """Convenience facade for steps 5-7 when text records already exist."""

    tokenized = tokenize_records(records, tokenizer)
    windowed = build_token_windows(tokenized, tokenizer, config=config)
    return build_model_tensors(windowed, tokenizer, device=device)

__all__ = [
    "ArticleText",
    "FinBertInputConfig",
    "FinBertTextRecord",
    "ModelReadyInputs",
    "PreparedArticleInputs",
    "build_model_tensors",
    "build_text_records",
    "build_token_windows",
    "extract_article_text",
    "load_tokenizer",
    "prepare_article_for_finbert",
    "prepare_model_inputs",
    "split_article_clauses",
    "split_article_sentences",
    "tokenize_records",
]
