"""UTF-8-safe Server-Sent Events framing and field parsing."""

import codecs
import re
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from typing import Optional

_LINE_ENDING_RE = re.compile(r"\r\n|\r|\n")
_EVENT_BOUNDARY_RE = re.compile(
    r"(?:(?:\r\n|(?<!\r)\n|\r(?!\n))){2,}"
)


@dataclass(frozen=True)
class SSEFrame:
    """Parsed fields from one SSE event frame."""

    data: Optional[str]
    event: Optional[str]
    event_id: Optional[str]
    retry: Optional[int]
    comments: tuple[str, ...]


def _canonical_frame(frame: str) -> str:
    return _LINE_ENDING_RE.sub("\n", frame) + "\n\n"


class IncrementalUTF8Decoder:
    """Decode arbitrary UTF-8 byte chunks without corrupting split code points."""

    def __init__(self):
        decoder_type = codecs.getincrementaldecoder("utf-8-sig")
        self._decoder = decoder_type(errors="replace")
        self._finalized = False

    def decode(self, chunk: bytes, *, final: bool = False) -> str:
        if self._finalized:
            raise RuntimeError("UTF-8 decoder already finalized")
        self._finalized = final
        return self._decoder.decode(chunk, final=final)


class IncrementalSSEDecoder:
    """Decode UTF-8 chunks and emit complete, LF-normalized SSE frames."""

    def __init__(self):
        self._decoder = IncrementalUTF8Decoder()
        self._buffer = ""
        self._finalized = False

    def feed(self, chunk: bytes | str) -> list[str]:
        if self._finalized:
            raise RuntimeError("SSE decoder already finalized")
        if isinstance(chunk, bytes):
            if chunk:
                self._buffer += self._decoder.decode(chunk)
        elif isinstance(chunk, str):
            if chunk:
                self._buffer += self._decoder.decode(chunk.encode("utf-8"))
        else:
            raise TypeError("SSE chunks must be bytes or str")
        return self._drain_complete_frames()

    def finalize(self) -> list[str]:
        if self._finalized:
            return []
        self._finalized = True
        self._buffer += self._decoder.decode(b"", final=True)
        frames = self._drain_complete_frames()
        if self._buffer.strip("\r\n"):
            # Upstreams commonly omit the final blank line. Preserve existing proxy
            # behavior by forwarding that last complete-looking frame at EOF.
            frames.append(_canonical_frame(self._buffer.rstrip("\r\n")))
        self._buffer = ""
        return frames

    def _drain_complete_frames(self) -> list[str]:
        frames = []
        while match := _EVENT_BOUNDARY_RE.search(self._buffer):
            frame = self._buffer[: match.start()]
            self._buffer = self._buffer[match.end() :]
            if frame:
                frames.append(_canonical_frame(frame))
        return frames


async def iter_sse_frames(chunks: AsyncIterable[bytes]) -> AsyncIterator[str]:
    """Yield complete SSE frames from arbitrary network byte chunks."""
    decoder = IncrementalSSEDecoder()
    async for chunk in chunks:
        for frame in decoder.feed(chunk):
            yield frame
    for frame in decoder.finalize():
        yield frame


def parse_sse_frame(frame: str) -> SSEFrame:
    """Parse standard SSE fields, joining repeated ``data`` lines with LF."""
    data_lines: list[str] = []
    event = None
    event_id = None
    retry = None
    comments = []

    normalized = _LINE_ENDING_RE.sub("\n", frame).rstrip("\n")
    for line in normalized.split("\n"):
        if line.startswith(":"):
            comments.append(line[1:])
            continue
        if not line:
            continue
        field, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if field == "data":
            data_lines.append(value)
        elif field == "event":
            event = value
        elif field == "id" and "\x00" not in value:
            event_id = value
        elif field == "retry" and value.isascii() and value.isdigit():
            retry = int(value)

    return SSEFrame(
        data="\n".join(data_lines) if data_lines else None,
        event=event,
        event_id=event_id,
        retry=retry,
        comments=tuple(comments),
    )


def replace_sse_data(frame: str, data: str) -> str:
    """Replace one frame's data fields while preserving comments and other fields."""
    normalized = _LINE_ENDING_RE.sub("\n", frame).rstrip("\n")
    output = []
    replaced = False
    for line in normalized.split("\n"):
        field = line.partition(":")[0]
        if field != "data":
            output.append(line)
            continue
        if not replaced:
            output.extend(f"data: {part}" for part in data.split("\n"))
            replaced = True
    if not replaced:
        output.extend(f"data: {part}" for part in data.split("\n"))
    return "\n".join(output) + "\n\n"
