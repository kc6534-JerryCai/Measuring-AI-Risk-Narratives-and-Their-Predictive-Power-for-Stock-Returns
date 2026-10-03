"""Step 7: pad token windows and build the tensors passed to FinBERT."""

from __future__ import annotations

import torch
from transformers import PreTrainedTokenizerBase

from .types import ModelReadyInputs, WindowedRecords


def build_model_tensors(
    windowed: WindowedRecords,
    tokenizer: PreTrainedTokenizerBase,
    *,
    device: torch.device | str | None = None,
) -> ModelReadyInputs:
    """Return model arguments and traceability metadata without calling a model."""

    features = [
        {
            "input_ids": list(window.input_ids),
            "attention_mask": list(window.attention_mask),
            "token_type_ids": list(window.token_type_ids),
            "special_tokens_mask": list(window.special_tokens_mask),
        }
        for window in windowed.windows
    ]
    if not features:
        raise ValueError("windowed must contain at least one token window.")
    padded = tokenizer.pad(features, padding=True, return_tensors="pt")
    special_tokens_mask = padded.pop("special_tokens_mask")
    expected_keys = {"input_ids", "attention_mask", "token_type_ids"}
    missing = expected_keys - set(padded)
    if missing:
        raise ValueError(f"Tokenizer output is missing model inputs: {sorted(missing)}")
    model_inputs = {key: padded[key] for key in sorted(expected_keys)}

    target_device = torch.device(device) if device is not None else None
    if target_device is not None:
        model_inputs = {
            key: value.to(target_device) for key, value in model_inputs.items()
        }
        special_tokens_mask = special_tokens_mask.to(target_device)
    metadata_device = target_device if target_device is not None else torch.device("cpu")

    return ModelReadyInputs(
        records=windowed.tokenized.records,
        model_inputs=model_inputs,
        special_tokens_mask=special_tokens_mask,
        window_record_indices=torch.tensor(
            [window.record_index for window in windowed.windows],
            dtype=torch.long,
            device=metadata_device,
        ),
        window_content_token_counts=torch.tensor(
            [window.content_token_count for window in windowed.windows],
            dtype=torch.long,
            device=metadata_device,
        ),
        record_token_counts=torch.tensor(
            [len(ids) for ids in windowed.tokenized.content_token_ids],
            dtype=torch.long,
            device=metadata_device,
        ),
    )
