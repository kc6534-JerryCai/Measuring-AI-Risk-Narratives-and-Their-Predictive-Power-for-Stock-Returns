"""Step 5: use the revision-pinned FinBERT tokenizer to create token IDs."""

from __future__ import annotations

from collections.abc import Sequence

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .types import FinBertInputConfig, FinBertTextRecord, TokenizedRecords


def load_tokenizer(
    config: FinBertInputConfig = FinBertInputConfig(),
) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(
        config.model_name,
        revision=config.model_revision,
    )


def tokenize_records(
    records: Sequence[FinBertTextRecord],
    tokenizer: PreTrainedTokenizerBase,
) -> TokenizedRecords:
    """Convert strings to vocabulary IDs without truncation or special tokens."""

    source_records = tuple(records)
    if not source_records:
        raise ValueError("records must contain at least one text record.")
    token_rows: list[tuple[int, ...]] = []
    for record in source_records:
        if not record.text.strip():
            raise ValueError(f"Record {record.record_id!r} has empty text.")
        token_ids = tuple(tokenizer.encode(record.text, add_special_tokens=False))
        if not token_ids:
            raise ValueError(f"Tokenizer produced no tokens for {record.record_id!r}.")
        token_rows.append(token_ids)
    return TokenizedRecords(source_records, tuple(token_rows))
