"""Step 6: create overlapping windows and add BERT special tokens."""

from __future__ import annotations

from transformers import PreTrainedTokenizerBase

from .types import FinBertInputConfig, TokenWindow, TokenizedRecords, WindowedRecords


def build_token_windows(
    tokenized: TokenizedRecords,
    tokenizer: PreTrainedTokenizerBase,
    *,
    config: FinBertInputConfig = FinBertInputConfig(),
) -> WindowedRecords:
    """Preserve all content IDs using the project's overlapping-window policy."""

    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError("The tokenizer must define CLS and SEP token IDs.")
    capacity = config.max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if capacity < 1:
        raise ValueError("max_length leaves no room for content tokens.")
    if config.window_overlap < 0 or config.window_overlap >= capacity:
        raise ValueError("window_overlap must be between 0 and capacity - 1.")

    step = capacity - config.window_overlap
    windows: list[TokenWindow] = []
    for record_index, content_ids in enumerate(tokenized.content_token_ids):
        for start in range(0, len(content_ids), step):
            content = content_ids[start : start + capacity]
            input_ids = (tokenizer.cls_token_id, *content, tokenizer.sep_token_id)
            windows.append(
                TokenWindow(
                    record_index=record_index,
                    content_token_count=len(content),
                    input_ids=input_ids,
                    attention_mask=(1,) * len(input_ids),
                    token_type_ids=(0,) * len(input_ids),
                    special_tokens_mask=(1, *((0,) * len(content)), 1),
                )
            )
            if start + capacity >= len(content_ids):
                break
    return WindowedRecords(tokenized=tokenized, windows=tuple(windows))
