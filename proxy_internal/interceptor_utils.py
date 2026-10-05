"""Shared matching and session-binding helpers for CLI interceptors."""

from typing import Optional



def content_matches_text(content, expected: str) -> bool:
    """Match direct text or text blocks in structured content."""
    if content == expected:
        return True
    if not isinstance(content, list):
        return False
    return any(
        block == expected
        or (isinstance(block, dict) and block.get("text") == expected)
        for block in content
    )


class SessionBindingMixin:
    """Explicitly bind and check request session identity."""

    session_id: Optional[str]

    def bind_session(self, session_id: Optional[str]) -> bool:
        if self.session_id is None:
            if not session_id:
                return True
            self.session_id = session_id
            return True
        return bool(session_id) and self.session_id == session_id
