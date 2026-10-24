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


def _synth_first_available(tts, text: str, candidates: list[str | None]):
    """Synthesize with the first voice that loads; None means the model default."""
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            if candidate is None:
                return tts.synth(text), None
            return tts.synth(text, voice=candidate), candidate
        except OSError as e:
            last_error = e
            print(
                f"[WARNING] [{_ts()}] [PHONON_TTS] Voice {candidate!r} could not "
                f"be loaded ({str(e)[:90]}); trying the next voice"
            )
    raise RuntimeError(f"No Phonon voice could be loaded: {last_error}")


def generate_audio_phonon(text: str, voice: str = "") -> tuple[np.ndarray, int]:
    """Generate mono PCM with Phonon and return it with its sample rate.

    `voice` is the catalog voice picked for this generation (see
    phonon_voices.select_voice). A cloned PHONON_VOICE_FILE always wins over it.
    """
    from audio_config import (
        PHONON_LANG,
        PHONON_MODEL_ID,
        PHONON_VOICE,
        PHONON_VOICE_FILE,
    )
    import ptts

    model_id = PHONON_MODEL_ID.strip()
    picked = voice.strip()
    configured = PHONON_VOICE.strip()
    voice_file = resolve_voice_file(PHONON_VOICE_FILE)
    rotating = voice_file is None and not configured and bool(picked)
    model_label = model_id or ptts.DEFAULT_MODEL
    if voice_file:
        voice_label = f"file:{voice_file.name}"
    else:
        voice_label = picked or configured or "model default"
    print(
        f"[INFO]    [{_ts()}] [PHONON_TTS] Loading model={model_label}, "
        f"lang={PHONON_LANG}, voice={voice_label}, device=auto"
    )
    tts = ptts.TTS(
        model=model_id or None,
        device=None,
        lang=PHONON_LANG,
        voice=None if (voice_file or rotating) else (configured or None),
    )
    if voice_file is not None:
        tts.add_voice(CUSTOM_VOICE_NAME, voice_file)
    print(
        f"[INFO]    [{_ts()}] [PHONON_TTS] Model ready: "
        f"device={tts.device}, sample_rate={tts.sample_rate}"
    )
    if voice_file:
        raw = tts.synth(text, voice=CUSTOM_VOICE_NAME)
    elif rotating:
        from phonon_voices import fallback_order

        raw, used = _synth_first_available(tts, text, fallback_order(picked))
        print(
            f"[INFO]    [{_ts()}] [PHONON_TTS] Voice used: "
            f"{used or 'model default'}"
        )
    else:
        raw = tts.synth(text)
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
