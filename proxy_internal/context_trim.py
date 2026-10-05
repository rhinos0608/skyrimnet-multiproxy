"""Recovering from upstream "context window exceeded" rejections.

A long SkyrimNet conversation can outgrow a model's context window; the upstream
answers HTTP 400 with a ``context_length_exceeded`` error rather than producing a
reply. Instead of letting the NPC fall silent, the caller drops the oldest
droppable history and retries (see ``proxy_internal.providers.zai``).

Ported from the community proxy that shipped this behaviour first; the message
list here is the provider-facing one, so ``messages[0]`` is often the system
prompt rather than the first user turn.
"""

import json
from typing import Optional

CONTEXT_EXCEEDED_CODE = "context_length_exceeded"

# Rough characters-per-token for English prose. Only used to size how much
# history to drop — the retry either fits or reports the new overage.
_CHARS_PER_TOKEN = 4.5

# How many trim-and-retry rounds a single request may use before the original
# error is surfaced. Each round re-measures against the upstream's own token
# count, so more than one round is only needed when the estimate undershoots.
MAX_TRIM_ROUNDS = 2


class ContextExceededError(Exception):
    """Upstream rejected the payload for exceeding the model's context window."""

    def __init__(self, excess_tokens: int):
        self.excess_tokens = max(0, int(excess_tokens))
        super().__init__(f"context window exceeded by {self.excess_tokens} tokens")


def _as_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def context_excess_tokens(status: int, body: str) -> Optional[int]:
    """Return the token overage of a ``context_length_exceeded`` rejection.

    ``None`` means the response is not a context-window rejection (or does not
    carry the token counts needed to size a trim), and the caller should handle
    it as a normal upstream error.
    """
    if status != 400 or CONTEXT_EXCEEDED_CODE not in body:
        return None
    try:
        error = json.loads(body).get("error")
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(error, dict) or error.get("code") != CONTEXT_EXCEEDED_CODE:
        return None
    counts = [
        _as_int(error.get("input_tokens")),
        _as_int(error.get("output_tokens")),
        _as_int(error.get("max_context_tokens")),
    ]
    if any(count is None for count in counts):
        return None
    overage = counts[0] + counts[1] - counts[2]
    return overage if overage > 0 else None


def message_chars(message: dict) -> int:
    """Character count of one message's content (str or multimodal list)."""
    content = message.get("content")
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(
            len(part["text"])
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return 0


def trim_messages_for_context(messages: list, excess_tokens: int) -> tuple[list, int]:
    """Drop the oldest droppable messages until ``excess_tokens`` are freed.

    Protected: the first message (SkyrimNet's diary/scenario context), the final
    message (the turn being answered), and every ``system`` message. Returns the
    new list plus the number of characters removed — ``0`` means nothing could be
    dropped, so the caller must surface the upstream error instead of retrying.
    """
    target_chars = int(max(0, excess_tokens) * _CHARS_PER_TOKEN)
    trimmed = list(messages)
    removed = 0
    while removed < target_chars:
        index = next(
            (
                i
                for i in range(1, len(trimmed) - 1)
                if trimmed[i].get("role") != "system"
            ),
            None,
        )
        if index is None:
            break
        removed += message_chars(trimmed.pop(index))
    return trimmed, removed
