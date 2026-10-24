"""Phonon TTS backend using Gradium's Rust runtime."""

from datetime import datetime
from pathlib import Path

import numpy as np

CUSTOM_VOICE_NAME = "custom"


def resolve_voice_file(raw: str) -> Path | None:
    """Return the configured custom voice file, or None when unset.

    Raises FileNotFoundError when a path is configured but missing, so a typo
    never silently falls back to a different voice.
    """
    raw = raw.strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"PHONON_VOICE_FILE does not exist: {path}")
    return path


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def generate_audio_phonon(text: str) -> tuple[np.ndarray, int]:
    """Generate mono PCM with Phonon and return it with its sample rate."""
    from audio_config import (
        PHONON_LANG,
        PHONON_MODEL_ID,
        PHONON_VOICE,
        PHONON_VOICE_FILE,
    )
    import ptts

    model_id = PHONON_MODEL_ID.strip()
    voice = PHONON_VOICE.strip()
    voice_file = resolve_voice_file(PHONON_VOICE_FILE)
    model_label = model_id or ptts.DEFAULT_MODEL
    voice_label = f"file:{voice_file.name}" if voice_file else (voice or "model default")
    print(
        f"[INFO]    [{_ts()}] [PHONON_TTS] Loading model={model_label}, "
        f"lang={PHONON_LANG}, voice={voice_label}, device=auto"
    )
    tts = ptts.TTS(
        model=model_id or None,
        device=None,
        lang=PHONON_LANG,
        voice=None if voice_file else (voice or None),
    )
    if voice_file is not None:
        tts.add_voice(CUSTOM_VOICE_NAME, voice_file)
    print(
        f"[INFO]    [{_ts()}] [PHONON_TTS] Model ready: "
        f"device={tts.device}, sample_rate={tts.sample_rate}"
    )
    raw = tts.synth(text, voice=CUSTOM_VOICE_NAME) if voice_file else tts.synth(text)
    audio = np.asarray(raw, dtype=np.float32)
    sample_rate = int(tts.sample_rate)

    if audio.ndim != 1 or audio.size == 0:
        raise ValueError(
            "Phonon returned invalid audio; expected non-empty mono PCM."
        )
    if sample_rate <= 0:
        raise ValueError(
            f"Phonon returned an invalid sample rate: {sample_rate}."
        )

    print(
        f"[INFO]    [{_ts()}] [PHONON_TTS] Audio generated: "
        f"{audio.size} samples, "
        f"{audio.size / sample_rate:.2f}s at {sample_rate} Hz"
    )
    return audio, sample_rate
