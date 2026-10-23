"""Phonon TTS backend using Gradium's Rust runtime."""

from datetime import datetime

import numpy as np


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def generate_audio_phonon(text: str) -> tuple[np.ndarray, int]:
    """Generate mono PCM with Phonon and return it with its sample rate."""
    from audio_config import PHONON_LANG, PHONON_MODEL_ID, PHONON_VOICE
    import ptts

    model_id = PHONON_MODEL_ID.strip()
    voice = PHONON_VOICE.strip()
    model_label = model_id or ptts.DEFAULT_MODEL
    voice_label = voice or "model default"
    print(
        f"[INFO]    [{_ts()}] [PHONON_TTS] Loading model={model_label}, "
        f"lang={PHONON_LANG}, voice={voice_label}, device=auto"
    )
    tts = ptts.TTS(
        model=model_id or None,
        device=None,
        lang=PHONON_LANG,
        voice=voice or None,
    )
    print(
        f"[INFO]    [{_ts()}] [PHONON_TTS] Model ready: "
        f"device={tts.device}, sample_rate={tts.sample_rate}"
    )
    audio = np.asarray(tts.synth(text), dtype=np.float32)
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
