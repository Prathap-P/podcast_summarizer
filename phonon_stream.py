"""Live Phonon speech: sentence-chunked streaming with a resident model."""

import re
import threading
from datetime import datetime
from typing import Iterator

import numpy as np

MAX_CHUNK_CHARS = 300

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")
_SOFT_BREAK = re.compile(r"(?<=[,;:])\s+")

_tts = None
_voice: str | None = None
_tts_lock = threading.Lock()
_stream_lock = threading.Lock()


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def split_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text into sentence-aligned pieces of at most max_chars.

    Short pieces keep time-to-first-audio low for any total text size.
    """
    pieces: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        sentence = sentence.strip()
        if sentence:
            pieces.extend(_split_long(sentence, max_chars))

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 1 + len(piece) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    return chunks


def _split_long(sentence: str, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]
    out: list[str] = []
    current = ""
    for part in _SOFT_BREAK.split(sentence):
        for group in _split_words(part, max_chars):
            if current and len(current) + 1 + len(group) > max_chars:
                out.append(current)
                current = group
            else:
                current = f"{current} {group}".strip()
    if current:
        out.append(current)
    return out


def _split_words(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    out: list[str] = []
    current = ""
    for word in text.split():
        while len(word) > max_chars:
            if current:
                out.append(current)
                current = ""
            out.append(word[:max_chars])
            word = word[max_chars:]
        if current and len(current) + 1 + len(word) > max_chars:
            out.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        out.append(current)
    return out


def _get_tts():
    """Load Phonon once and keep it resident for later requests."""
    global _tts, _voice
    with _tts_lock:
        if _tts is None:
            from audio_config import (
                PHONON_LANG,
                PHONON_MODEL_ID,
                PHONON_VOICE,
                PHONON_VOICE_FILE,
            )
            from phonon_tts import CUSTOM_VOICE_NAME, resolve_voice_file
            import ptts

            voice_file = resolve_voice_file(PHONON_VOICE_FILE)
            print(f"[INFO]    [{_ts()}] [PHONON_STREAM] Loading resident model")
            tts = ptts.TTS(
                model=PHONON_MODEL_ID.strip() or None,
                device=None,
                lang=PHONON_LANG,
                voice=None if voice_file else (PHONON_VOICE.strip() or None),
            )
            if voice_file is not None:
                tts.add_voice(CUSTOM_VOICE_NAME, voice_file)
                _voice = CUSTOM_VOICE_NAME
                print(
                    f"[INFO]    [{_ts()}] [PHONON_STREAM] Custom voice: "
                    f"{voice_file.name}"
                )
            _tts = tts
            print(
                f"[INFO]    [{_ts()}] [PHONON_STREAM] Ready: "
                f"device={_tts.device}, sample_rate={_tts.sample_rate}"
            )
        return _tts


def get_sample_rate() -> int:
    """Return the stream sample rate, loading the model if needed."""
    return int(_get_tts().sample_rate)


def to_pcm16(chunk: np.ndarray) -> bytes:
    """Convert a float32 chunk in [-1, 1] to little-endian int16 bytes."""
    samples = np.clip(np.asarray(chunk, dtype=np.float32), -1.0, 1.0)
    return (samples * 32767.0).astype("<i2").tobytes()


def stream_pcm16(text: str) -> Iterator[bytes]:
    """Yield int16 PCM bytes for text, one sentence group at a time.

    Streams are serialised: Phonon runs one generation at a time. Closing the
    iterator early (client disconnect or Stop) stops generation.
    """
    tts = _get_tts()
    pieces = split_text(text)
    with _stream_lock:
        for index, piece in enumerate(pieces, start=1):
            print(
                f"[DEBUG]   [PHONON_STREAM] Chunk {index}/{len(pieces)} "
                f"({len(piece)} chars)"
            )
            stream_kwargs = {"voice": _voice} if _voice else {}
            with tts.stream(piece, **stream_kwargs) as audio:
                for chunk in audio:
                    data = to_pcm16(chunk)
                    if data:
                        yield data
