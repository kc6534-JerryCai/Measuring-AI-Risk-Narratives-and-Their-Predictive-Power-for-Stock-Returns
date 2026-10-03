"""Shared data structures for the preprocessing steps."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.finbert_vectorization import DEFAULT_MODEL, DEFAULT_REVISION
from src.preprocess_financial_news import ClauseRecord, SentenceRecord


@dataclass(frozen=True)
class FinBertInputConfig:
    model_name: str = DEFAULT_MODEL
    model_revision: str = DEFAULT_REVISION
    max_length: int = 512
    window_overlap: int = 64


@dataclass(frozen=True)
class CleanedArticle:
    document_id: str
    title: str
    cleaned_document_text: str


@dataclass(frozen=True)
class SentenceArticle:
    document_id: str
    title: str
    cleaned_document_text: str
    sentences: tuple[SentenceRecord, ...]
    parsed_sentences: tuple[object, ...]


@dataclass(frozen=True)
class ArticleText:
    document_id: str
    title: str
    cleaned_document_text: str
    sentences: tuple[SentenceRecord, ...]
    clauses: tuple[ClauseRecord, ...]


@dataclass(frozen=True)
class FinBertTextRecord:
    record_id: str
    document_id: str
    kind: str
    text: str


@dataclass(frozen=True)
class TokenizedRecords:
    records: tuple[FinBertTextRecord, ...]
    content_token_ids: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class TokenWindow:
    record_index: int
    content_token_count: int
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    token_type_ids: tuple[int, ...]
    special_tokens_mask: tuple[int, ...]


@dataclass(frozen=True)
class WindowedRecords:
    tokenized: TokenizedRecords
    windows: tuple[TokenWindow, ...]


@dataclass(frozen=True)
class ModelReadyInputs:
    """Final tensors immediately before the ``FinBERT(**model_inputs)`` call."""

    records: tuple[FinBertTextRecord, ...]
    model_inputs: dict[str, torch.Tensor]
    special_tokens_mask: torch.Tensor
    window_record_indices: torch.Tensor
    window_content_token_counts: torch.Tensor
    record_token_counts: torch.Tensor

    @property
    def window_count(self) -> int:
        return int(self.window_record_indices.numel())
