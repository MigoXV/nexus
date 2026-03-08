from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_SID_OTHERS_PATTERN = re.compile(r"^\s*<sid\s+<others>>\s*(.*)$", re.DOTALL)
_SID_SPEAKER_PATTERN = re.compile(
    r"^\s*<sid\s+([^\s<>]+)\s+([0-9]+(?:\.[0-9]+)?)>\s*(.*)$",
    re.DOTALL,
)
_WHITESPACE_PATTERN = re.compile(r"\s+")
_CJK_SPACE_PATTERN = re.compile(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])")
_TTS_DELIMITER_SPACING_PATTERN = re.compile(r"\s*([。！？；])\s*")

_SENTENCE_DELIMITER_MAP = {
    ".": "。",
    "!": "！",
    "?": "？",
    ";": "；",
    "。": "。",
    "！": "！",
    "？": "？",
    "；": "；",
}
_TTS_SENTENCE_DELIMITERS = set(_SENTENCE_DELIMITER_MAP)

_EMOJI_RANGES = (
    (0x1F300, 0x1F5FF),
    (0x1F600, 0x1F64F),
    (0x1F680, 0x1F6FF),
    (0x1F700, 0x1F77F),
    (0x1F780, 0x1F7FF),
    (0x1F800, 0x1F8FF),
    (0x1F900, 0x1F9FF),
    (0x1FA70, 0x1FAFF),
    (0x2600, 0x26FF),
    (0x2700, 0x27BF),
    (0xFE00, 0xFE0F),
)


@dataclass(frozen=True)
class PreparedRealtimeUserTurn:
    raw_transcript: str
    display_transcript: str
    model_text: str
    speaker_name: str | None = None


@dataclass
class SanitizedModelOutputAccumulator:
    raw_text: str = ""
    display_text: str = ""
    tts_text: str = ""

    def push(self, delta: str) -> tuple[str, str]:
        if not delta:
            return "", ""

        self.raw_text += delta
        next_display = sanitize_model_output_for_display(self.raw_text)
        next_tts = sanitize_model_output_for_tts(self.raw_text)

        display_delta = _incremental_suffix(self.display_text, next_display)
        tts_delta = _incremental_suffix(self.tts_text, next_tts)

        self.display_text = next_display
        self.tts_text = next_tts
        return display_delta, tts_delta


def prepare_realtime_user_turn(transcript: str) -> PreparedRealtimeUserTurn:
    speaker_name, display_transcript = parse_asr_speaker_prefix(transcript)
    if speaker_name:
        model_text = (
            f"当前说话人是{speaker_name}。"
            "这只是辅助上下文，不要直接复述说话人标签。"
            f"用户说：{display_transcript}"
        )
    else:
        model_text = display_transcript

    return PreparedRealtimeUserTurn(
        raw_transcript=transcript,
        display_transcript=display_transcript,
        model_text=model_text,
        speaker_name=speaker_name,
    )


def parse_asr_speaker_prefix(transcript: str) -> tuple[str | None, str]:
    if not transcript:
        return None, ""

    others_match = _SID_OTHERS_PATTERN.match(transcript)
    if others_match:
        return None, others_match.group(1).strip()

    speaker_match = _SID_SPEAKER_PATTERN.match(transcript)
    if speaker_match:
        return speaker_match.group(1), speaker_match.group(3).strip()

    return None, transcript


def sanitize_model_output_for_display(text: str) -> str:
    return _sanitize_model_output(text, preserve_sentence_delimiters=False)


def sanitize_model_output_for_tts(text: str) -> str:
    return _sanitize_model_output(text, preserve_sentence_delimiters=True)


def _sanitize_model_output(text: str, *, preserve_sentence_delimiters: bool) -> str:
    if not text:
        return ""

    chunks: list[str] = []
    for char in text:
        if _is_emoji(char):
            continue
        if char in {"\r", "\t"}:
            char = " "
        elif char == "\n":
            char = "。" if preserve_sentence_delimiters else " "

        if char.isspace():
            chunks.append(" ")
            continue

        if preserve_sentence_delimiters and char in _TTS_SENTENCE_DELIMITERS:
            chunks.append(_SENTENCE_DELIMITER_MAP[char])
            continue

        category = unicodedata.category(char)
        if category.startswith(("P", "S")):
            continue

        chunks.append(char)

    result = "".join(chunks)
    result = _WHITESPACE_PATTERN.sub(" ", result)
    result = _CJK_SPACE_PATTERN.sub("", result)
    if preserve_sentence_delimiters:
        result = _TTS_DELIMITER_SPACING_PATTERN.sub(r"\1", result)
    return result.strip()


def _is_emoji(char: str) -> bool:
    codepoint = ord(char)
    return any(start <= codepoint <= end for start, end in _EMOJI_RANGES)


def _incremental_suffix(previous: str, current: str) -> str:
    if not current:
        return ""
    if not previous:
        return current
    if current.startswith(previous):
        return current[len(previous):]

    prefix_len = 0
    max_prefix = min(len(previous), len(current))
    while prefix_len < max_prefix and previous[prefix_len] == current[prefix_len]:
        prefix_len += 1
    return current[prefix_len:]
