"""Conservative response-markup normalization for SkyrimNet dialogue."""

import json
import logging
import re
import time
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from contextlib import suppress
from typing import Optional

from proxy_internal.config_store import _load_config
from proxy_internal.sse_parser import parse_sse_frame, replace_sse_data

logger = logging.getLogger("proxy")

_TAG_JUNK = "*_`~.,:;!?\"'|+=#\\/ \t-"
_EMPHASIS = "*_`~"
_TAG_BRACKETS = {"<": ">", "[": "]", "(": ")", "{": "}"}
_TAG_CLOSERS = {closer: opener for opener, closer in _TAG_BRACKETS.items()}
_TAG_PREFIX_RE = re.compile(
    r"\[?/?(?:[A-Za-z][A-Za-z0-9]*(?:[ _.\-]+[A-Za-z0-9]*){0,2})?[\]>)}]{0,2}\Z"
)
_BRACKET_SCAN = {
    opener: re.compile(f"[{re.escape(opener)}{re.escape(closer)}]")
    for opener, closer in _TAG_BRACKETS.items()
}
_TAG_CANDIDATE_RE = re.compile(
    r"(?P<pre>[*_`~]{0,3})(?P<open>[<\[])(?P<inner>[^<>\[\]\n]{1,64})"
    r"(?P<close>[>\]])(?P<post>[*_`~]{0,3})"
)
_WELL_FORMED_RE = re.compile(r"</?([A-Za-z][A-Za-z0-9_]*)/?>")
_CLEAN_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_LOOSE_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:[ _.\-][A-Za-z0-9]+)*$")


def _default_known_tags() -> tuple[str, ...]:
    """Return configured known tags, evaluated lazily at use time."""
    configured = _load_config().get("known_tags", ["internal_thought"])
    if not isinstance(configured, list):
        return ("internal_thought",)
    return tuple(str(tag) for tag in configured[:64])


def _tag_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


class _TagVocabulary:
    """Known tag spellings scoped to one response or stream."""

    def __init__(self, known_tags: Iterable[str]):
        self._tags: dict[str, str] = {}
        for tag in known_tags:
            self.remember(tag)

    def remember(self, name: str) -> None:
        key = _tag_key(name)
        if key and key not in self._tags and len(self._tags) < 64:
            self._tags[key] = name

    def resolve(self, name: str) -> Optional[str]:
        return self._tags.get(_tag_key(name))


def _repair_tags(
    text: str, vocabulary: _TagVocabulary
) -> tuple[str, list[str], bool]:
    """Normalize generic tag delimiters without changing enclosed content."""
    repairs: list[str] = []

    def _fix(match: re.Match) -> str:
        raw = match.group(0)
        inner = match.group("inner")
        core = inner.strip(_TAG_JUNK)
        closing = inner.strip(_EMPHASIS + " \t").startswith("/")
        self_closing = inner.strip(_EMPHASIS + " \t").endswith("/")
        if not core or not _LOOSE_NAME_RE.match(core):
            return raw

        name = vocabulary.resolve(core)
        square = match.group("open") != "<" or match.group("close") != ">"
        if name is None:
            if square or not _CLEAN_NAME_RE.match(core):
                return raw
            name = core

        fixed = f"<{'/' if closing else ''}{name}{'/' if self_closing else ''}>"
        vocabulary.remember(name)
        if fixed == raw:
            return raw
        repairs.append(f"{raw}  ->  {fixed}")
        return fixed

    output = _TAG_CANDIDATE_RE.sub(_fix, text)
    return output, repairs, bool(repairs) and output.endswith(">")


_VOICE_TAGS = {
    "EMOTION": (
        "AFFECTION", "ANGER", "AROUSAL", "AWE", "BITTERNESS", "CONFUSION",
        "CONTEMPLATION", "CONTENTMENT", "DETERMINATION", "DISGUST", "ENTHUSIASM",
        "FEAR", "HELPLESSNESS", "LONGING", "PRIDE", "RELIEF", "SADNESS", "SHAME",
        "SURPRISE",
    ),
    "STYLE": ("WHISPERING", "SHOUTING", "SINGING"),
    "SFX": (
        "LAUGHTER", "SIGH", "COUGH", "CRYING", "SCREAMING", "SNIFF", "SNEEZE",
        "HUMMING",
    ),
}
_VOICE_SUFFIX = {name: category for category, names in _VOICE_TAGS.items() for name in names}
_VOICE_ALIAS = {
    "LAUGH": "LAUGHTER", "LAUGHS": "LAUGHTER", "LAUGHING": "LAUGHTER",
    "CHUCKLE": "LAUGHTER", "CHUCKLES": "LAUGHTER", "CHUCKLING": "LAUGHTER",
    "GIGGLE": "LAUGHTER", "GIGGLES": "LAUGHTER", "SNORT": "LAUGHTER",
    "SIGHS": "SIGH", "SIGHING": "SIGH", "COUGHS": "COUGH", "COUGHING": "COUGH",
    "CLEARSTHROAT": "COUGH", "CRY": "CRYING", "CRIES": "CRYING", "SOB": "CRYING",
    "SOBS": "CRYING", "SOBBING": "CRYING", "WEEP": "CRYING", "WEEPING": "CRYING",
    "SCREAM": "SCREAMING", "SCREAMS": "SCREAMING", "SHRIEK": "SCREAMING",
    "SHRIEKS": "SCREAMING", "SHRIEKING": "SCREAMING", "SNIFFS": "SNIFF",
    "SNIFFLE": "SNIFF", "SNIFFLES": "SNIFF", "SNIFFING": "SNIFF",
    "SNEEZES": "SNEEZE", "SNEEZING": "SNEEZE", "HUM": "HUMMING", "HUMS": "HUMMING",
    "WHISPER": "WHISPERING", "WHISPERS": "WHISPERING", "WHISPERED": "WHISPERING",
    "SHOUT": "SHOUTING", "SHOUTS": "SHOUTING", "YELL": "SHOUTING",
    "YELLS": "SHOUTING", "YELLING": "SHOUTING", "SING": "SINGING",
    "SINGS": "SINGING", "SUNG": "SINGING", "AFFECTIONATE": "AFFECTION",
    "ANGRY": "ANGER", "RAGE": "ANGER", "FURY": "ANGER", "FURIOUS": "ANGER",
    "AROUSED": "AROUSAL", "AWED": "AWE", "BITTER": "BITTERNESS",
    "CONFUSED": "CONFUSION", "CONTEMPLATIVE": "CONTEMPLATION",
    "THOUGHTFUL": "CONTEMPLATION", "CONTENT": "CONTENTMENT",
    "DETERMINED": "DETERMINATION", "DISGUSTED": "DISGUST",
    "ENTHUSIASTIC": "ENTHUSIASM", "EXCITED": "ENTHUSIASM",
    "EXCITEMENT": "ENTHUSIASM", "AFRAID": "FEAR", "SCARED": "FEAR",
    "FEARFUL": "FEAR", "TERRIFIED": "FEAR", "HELPLESS": "HELPLESSNESS",
    "LONGS": "LONGING", "YEARNING": "LONGING", "PROUD": "PRIDE",
    "RELIEVED": "RELIEF", "SAD": "SADNESS", "SORROW": "SADNESS",
    "ASHAMED": "SHAME", "SURPRISED": "SURPRISE", "SHOCK": "SURPRISE",
    "SHOCKED": "SURPRISE",
}
_VOICE_WRAPPED_RE = re.compile(
    r"(?P<pre>[*_`~]{0,3})(?P<open>\[{1,2}|<|\(|\{)(?P<slash>/?)"
    r"(?P<inner>[A-Za-z][A-Za-z0-9 _.\-]{0,39})(?:\]{1,2}|>|\)|\})"
    r"(?P<post>[*_`~]{0,3})"
)
_VOICE_ACTION_RE = re.compile(
    r"(?<![*\w])\*{1,2}(?P<word>[A-Za-z]{3,20})\*{1,2}(?![*\w])"
)


def _voice_key(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _resolve_voice(inner: str, *, allow_bare: bool) -> Optional[str]:
    tokens = [token for token in re.split(r"[^A-Za-z0-9]+", inner) if token]
    if not tokens or len(tokens) > 3:
        return None

    category = tokens[0].upper() if tokens[0].upper() in _VOICE_TAGS else None
    name = _voice_key("".join(tokens[1:] if category else tokens))
    if not name:
        return "" if category else None

    name = name if name in _VOICE_SUFFIX else _VOICE_ALIAS.get(name, "")
    if not name:
        return "" if category else None
    if not category and not allow_bare:
        return None
    return f"{_VOICE_SUFFIX[name]}-{name}"


def _voice_padded(match: re.Match, fixed: str) -> str:
    if not fixed:
        return ""
    source, start, end = match.string, match.start(), match.end()
    left = " " if start > 0 and source[start - 1] not in " \t\n([\"'" else ""
    right = " " if end < len(source) and source[end] not in " \t\n.,!?;:)]\"'" else ""
    return left + fixed + right


def _repair_voice_tags(text: str, previous: str = "") -> tuple[str, list[str], bool]:
    repairs: list[str] = []
    dropped = False
    previous = (
        previous
        if previous and previous not in _EMPHASIS + " \t\n" + "".join(_TAG_BRACKETS)
        else ""
    )
    text = previous + text

    def _fix_wrapped(match: re.Match) -> str:
        nonlocal dropped
        raw, inner = match.group(0), match.group("inner")
        canonical = _resolve_voice(
            inner,
            allow_bare=match.group("open")[0] in "[<" or inner.isupper(),
        )
        if canonical is None:
            return raw
        if match.group("slash"):
            canonical = ""
        fixed = _voice_padded(match, f"[{canonical}]" if canonical else "")
        if fixed == raw:
            return raw
        dropped = dropped or not canonical
        repairs.append(f"{raw}  ->  {fixed.strip() or '(dropped)'}")
        return fixed

    def _fix_action(match: re.Match) -> str:
        raw = match.group(0)
        canonical = _resolve_voice(match.group("word"), allow_bare=True)
        if not canonical or not canonical.startswith("SFX-"):
            return raw
        fixed = _voice_padded(match, f"[{canonical}]")
        repairs.append(f"{raw}  ->  {fixed.strip()}")
        return fixed

    output = _VOICE_WRAPPED_RE.sub(_fix_wrapped, text)
    if "*" in output:
        output = _VOICE_ACTION_RE.sub(_fix_action, output)
    if dropped:
        output = re.sub(r"[ \t]{2,}", " ", output)
        output = re.sub(r"[ \t]+([\n.,!?;:])", r"\1", output)
        output = re.sub(r"(?m)^[ \t]+", "", output)
    return output[len(previous):], repairs, bool(repairs) and output.endswith("]")


def _tag_counts(text: str) -> dict[str, list[int]]:
    counts: dict[str, list[int]] = {}
    for match in _WELL_FORMED_RE.finditer(text):
        tag, name = match.group(0), match.group(1)
        if tag.endswith("/>"):
            continue
        slot = counts.setdefault(name, [0, 0])
        slot[1 if tag.startswith("</") else 0] += 1
    return counts


def _balance_warnings(
    counts: dict[str, list[int]], vocabulary: _TagVocabulary
) -> list[str]:
    warnings = []
    for name, (opens, closes) in counts.items():
        if opens != closes and vocabulary.resolve(name) is not None:
            warnings.append(f"unbalanced <{name}>: {opens} open / {closes} close")
    return warnings


def _summarize_examples(items: list[str], limit: int = 5) -> str:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    examples = [
        f"{item} (x{count})" if count > 1 else item
        for item, count in list(counts.items())[:limit]
    ]
    omitted = len(counts) - len(examples)
    if omitted:
        examples.append(f"... {omitted} more unique")
    return "  |  ".join(examples)


def _log_repairs(
    request_id: str, model: str, repairs: list[str], warnings: list[str]
) -> None:
    if repairs:
        logger.warning(
            f"[{request_id}] {model} markup repaired x{len(repairs)}: "
            + _summarize_examples(repairs)
        )
    if warnings:
        logger.warning(
            f"[{request_id}] {model} markup warnings x{len(warnings)}: "
            + _summarize_examples(warnings)
        )


def sanitize_response(
    text: str,
    request_id: str,
    model: str,
    *,
    known_tags: Optional[Iterable[str]] = None,
) -> str:
    """Repair markup in one complete OpenAI-compatible response."""
    if not text or not any(char in text for char in "<[({*"):
        return text

    vocabulary = _TagVocabulary(known_tags or _default_known_tags())
    fixed, repairs, _ = _repair_voice_tags(text)
    if "<" in fixed or "[" in fixed:
        fixed, tag_repairs, _ = _repair_tags(fixed, vocabulary)
        repairs.extend(tag_repairs)
    _log_repairs(
        request_id,
        model,
        repairs,
        _balance_warnings(_tag_counts(fixed), vocabulary),
    )
    return fixed


class StreamSanitizer:
    """Apply response repairs while holding only a potentially incomplete tag tail."""

    MAX_HOLD = 96
    ACTION_HOLD = 32

    def __init__(
        self,
        request_id: str,
        model: str,
        *,
        known_tags: Optional[Iterable[str]] = None,
    ):
        self.request_id = request_id
        self.model = model
        self.pending = ""
        self.counts: dict[str, list[int]] = {}
        self.repairs: list[str] = []
        self.drop_lead_emphasis = False
        self.last_char = ""
        self.vocabulary = _TagVocabulary(known_tags or _default_known_tags())

    def feed(self, text: str) -> str:
        self.pending += text
        cut = self._safe_cut(self.pending)
        head, self.pending = self.pending[:cut], self.pending[cut:]
        return self._emit(head)

    def flush(self) -> str:
        output = self._emit(self.pending)
        self.pending = ""
        _log_repairs(
            self.request_id,
            self.model,
            self.repairs,
            _balance_warnings(self.counts, self.vocabulary),
        )
        self.repairs = []
        self.counts = {}
        return output

    def _safe_cut(self, buffer: str) -> int:
        hold = len(buffer)
        for opener, closer in _TAG_BRACKETS.items():
            if opener not in buffer:
                continue
            index = self._unclosed(buffer, opener, closer)
            if (
                index is not None
                and len(buffer) - index <= self.MAX_HOLD
                and self._tag_shaped(buffer, index)
                and not _TAG_CANDIDATE_RE.match(buffer, index)
            ):
                hold = min(hold, index)
        if buffer and buffer[-1] in _TAG_CLOSERS:
            index = self._matching_open(buffer)
            if (
                index is not None
                and len(buffer) - index <= self.MAX_HOLD
                and self._tag_shaped(buffer, index)
            ):
                hold = min(hold, index)
        index = self._open_emphasis(buffer)
        if index is not None and len(buffer) - index <= self.ACTION_HOLD:
            hold = min(hold, index)
        while hold > 0 and buffer[hold - 1] in _EMPHASIS and len(buffer) - hold < 3:
            hold -= 1
        return hold

    @staticmethod
    def _tag_shaped(buffer: str, index: int) -> bool:
        return buffer[index] == "<" or bool(_TAG_PREFIX_RE.match(buffer, index + 1))

    @classmethod
    def _unclosed(cls, buffer: str, opener: str, closer: str) -> Optional[int]:
        base = max(0, len(buffer) - cls.MAX_HOLD - 1)
        depth, start = 0, None
        for match in _BRACKET_SCAN[opener].finditer(buffer, base):
            if match.group() == opener:
                if depth == 0:
                    start = match.start()
                depth += 1
            elif depth:
                depth -= 1
        return start if depth else None

    @staticmethod
    def _matching_open(buffer: str) -> Optional[int]:
        closer, opener, depth = buffer[-1], _TAG_CLOSERS[buffer[-1]], 0
        for index in range(len(buffer) - 1, -1, -1):
            depth += (buffer[index] == closer) - (buffer[index] == opener)
            if depth == 0:
                return index
        return None

    @staticmethod
    def _open_emphasis(buffer: str) -> Optional[int]:
        if "*" not in buffer:
            return None
        runs = [match.span() for match in re.finditer(r"\*+", buffer)]
        if runs[-1][1] == len(buffer):
            return runs[-2][0] if len(runs) % 2 == 0 else runs[-1][0]
        return runs[-1][0] if len(runs) % 2 else None

    def _emit(self, chunk: str) -> str:
        if not chunk:
            return chunk
        if self.drop_lead_emphasis:
            self.drop_lead_emphasis = False
            stripped = chunk.lstrip(_EMPHASIS)
            if 0 < len(chunk) - len(stripped) <= 3:
                chunk = stripped
            if not chunk:
                return chunk

        ends_on_voice_tag = False
        if any(char in chunk for char in "<[({*"):
            chunk, voice_repairs, ends_on_voice_tag = _repair_voice_tags(
                chunk, self.last_char
            )
            self.repairs.extend(voice_repairs)
            if not chunk:
                self.drop_lead_emphasis = ends_on_voice_tag
                return chunk

        if "<" in chunk or "[" in chunk:
            fixed, repairs, ends_on_tag = _repair_tags(chunk, self.vocabulary)
        else:
            fixed, repairs, ends_on_tag = chunk, [], False
        self.drop_lead_emphasis = ends_on_tag or ends_on_voice_tag
        self.repairs.extend(repairs)
        if "<" in fixed:
            for name, (opens, closes) in _tag_counts(fixed).items():
                slot = self.counts.setdefault(name, [0, 0])
                slot[0] += opens
                slot[1] += closes
        if fixed:
            self.last_char = fixed[-1]
        return fixed


def _text_chunk(
    request_id: str,
    model: str,
    text: str,
    template: Optional[dict],
    choice_index: int,
    frame_template: Optional[str] = None,
) -> str:
    base = {key: value for key, value in (template or {}).items() if key != "choices"}
    base.setdefault("id", f"chatcmpl-{request_id}")
    base.setdefault("object", "chat.completion.chunk")
    base.setdefault("created", int(time.time()))
    base.setdefault("model", model)
    base["choices"] = [
        {"index": choice_index, "delta": {"content": text}, "finish_reason": None}
    ]
    payload = json.dumps(base)
    return (
        replace_sse_data(frame_template, payload)
        if frame_template is not None
        else f"data: {payload}\n\n"
    )


async def sanitize_sse_stream(
    events: AsyncIterable[str | bytes], request_id: str, model: str
) -> AsyncIterator[str | bytes]:
    """Repair every ``choice.delta.content`` in OpenAI-shaped SSE events."""
    sanitizers: dict[int, StreamSanitizer] = {}
    last_templates: dict[int, tuple[dict, str]] = {}

    def _sanitizer(choice_index: int) -> StreamSanitizer:
        sanitizer = sanitizers.get(choice_index)
        if sanitizer is None:
            sanitizer = StreamSanitizer(request_id, model)
            sanitizers[choice_index] = sanitizer
        return sanitizer

    def _flush_chunks():
        for choice_index, sanitizer in sanitizers.items():
            tail = sanitizer.flush()
            if tail:
                template, frame_template = last_templates.get(
                    choice_index, (None, None)
                )
                yield _text_chunk(
                    request_id,
                    model,
                    tail,
                    template,
                    choice_index,
                    frame_template,
                )

    iterator = events.__aiter__()
    try:
        async for event in iterator:
            decoded = event.decode("utf-8", errors="replace") if isinstance(event, bytes) else event
            payload = parse_sse_frame(decoded).data
            if payload is None:
                yield event
                continue
            if payload == "[DONE]":
                for tail_event in _flush_chunks():
                    yield tail_event
                yield event
                continue

            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                yield event
                continue

            choices = obj.get("choices") or []
            if not isinstance(choices, list):
                yield event
                continue

            emitted_choices = []
            modified = False
            leading_flushes = []
            for position, choice in enumerate(choices):
                if not isinstance(choice, dict):
                    emitted_choices.append(choice)
                    continue

                raw_index = choice.get("index", position)
                choice_index = raw_index if isinstance(raw_index, int) else position
                delta = choice.get("delta")
                content = delta.get("content") if isinstance(delta, dict) else None
                finish_reason = choice.get("finish_reason")

                if isinstance(content, str) and content:
                    sanitizer = _sanitizer(choice_index)
                    last_templates[choice_index] = (obj, decoded)
                    output = sanitizer.feed(content)
                    if finish_reason:
                        output += sanitizer.flush()
                    if not output:
                        modified = True
                        if finish_reason:
                            # Terminal choice with fully sanitized content:
                            # emit with empty content so finish_reason and
                            # top-level fields (usage, etc.) are preserved.
                            choice = dict(choice)
                            choice["delta"] = dict(delta)
                            choice["delta"]["content"] = ""
                            emitted_choices.append(choice)
                        continue
                    if output != content:
                        choice = dict(choice)
                        choice["delta"] = dict(delta)
                        choice["delta"]["content"] = output
                        modified = True
                    emitted_choices.append(choice)
                    continue

                sanitizer = sanitizers.get(choice_index)
                if finish_reason and sanitizer is not None:
                    tail = sanitizer.flush()
                    if tail:
                        leading_flushes.append(
                            _text_chunk(
                                request_id,
                                model,
                                tail,
                                obj,
                                choice_index,
                                decoded,
                            )
                        )
                emitted_choices.append(choice)

            for tail_event in leading_flushes:
                yield tail_event
            if modified:
                if emitted_choices:
                    obj["choices"] = emitted_choices
                    yield replace_sse_data(decoded, json.dumps(obj))
                continue
            yield event

        for tail_event in _flush_chunks():
            yield tail_event
    finally:
        close = getattr(iterator, "aclose", None)
        if close is not None:
            with suppress(Exception):
                await close()
        for sanitizer in sanitizers.values():
            sanitizer.pending = ""
