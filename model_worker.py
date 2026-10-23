"""Top-level picklable worker functions for subprocess-isolated model inference.

All model imports live INSIDE each function body so the main Flask process
never loads model weights. The OS reclaims all memory when the subprocess exits.
"""

import os
import time
from datetime import datetime

from dotenv import load_dotenv

from utils import format_elapsed

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def asr_worker(url: str, video_id: str) -> str:
    """Transcribe YouTube audio in an isolated subprocess.

    Reads ASR_BACKEND env var to dispatch to the correct backend.
    Returns the transcript string (or "Error: ..." on failure).
    """
    load_dotenv()
    started = time.perf_counter()
    try:
        backend = os.getenv("ASR_BACKEND", "whisper")
        print(f"[INFO]    [{_ts()}] [ASR subprocess] Starting, backend={backend}, video_id={video_id}")
        if backend == "qwen_omni":
            from qwen_omni_backend import get_transcript_via_qwen
            result = get_transcript_via_qwen(url, video_id)
        else:
            from whisper_transcriber import get_transcript_via_whisper
            result = get_transcript_via_whisper(url, video_id)
        print(f"[INFO]    [{_ts()}] [ASR subprocess] Complete, len={len(result)} chars")
        return result
    finally:
        elapsed = time.perf_counter() - started
        print(
            f"[TIME]    [{_ts()}] [ASR subprocess] "
            f"Transcription took: {format_elapsed(elapsed)}"
        )

def tts_worker(text: str, output_path: str, voice: str = "", video_id: str = "", video_title: str = "") -> str:
    """Generate TTS audio and save to file in an isolated subprocess.

    Reads TTS_BACKEND env var to dispatch to the correct backend.
    Always writes the .wav via kokoro_tts.create_audio_file() and returns
    the file path — the numpy audio array never crosses the queue boundary.

    Args:
        text: The spoken-word script to synthesise.
        output_path: Unused; kept for call-site compatibility.
        voice: Optional voice override for the voxtral backend.  Empty string
            means "use the backend default".
        video_id: YouTube video ID used for output filename (empty → timestamp fallback).
        video_title: YouTube video title used for output filename.
    """
    load_dotenv()
    started = time.perf_counter()
    backend = os.getenv("TTS_BACKEND", "kokoro")
    print(f"[INFO]    [{_ts()}] [TTS subprocess] Starting, backend={backend}, chars={len(text)}, voice={voice!r}, video_id={video_id!r}")
    try:
        return _generate_tts(text, backend, voice, video_id, video_title)
    finally:
        elapsed = time.perf_counter() - started
        print(
            f"[TIME]    [{_ts()}] [TTS subprocess] "
            f"Audio generation took: {format_elapsed(elapsed)}"
        )


def _generate_tts(text: str, backend: str, voice: str, video_id: str, video_title: str) -> str:
    """Dispatch TTS and write the wav. Split out so timing wraps the whole call."""
    if backend == "kokoro":
        os.environ["HF_HUB_OFFLINE"] = "1"
        print(f"[INFO]    [{_ts()}] [TTS subprocess] Kokoro Hugging Face access disabled; using local cache only")

    from kokoro_tts import create_audio_file
    if backend == "qwen_omni":
        from qwen_omni_backend import generate_audio_qwen as generate_audio
        audio = generate_audio(text)
        final_path = create_audio_file(audio, video_id=video_id, video_title=video_title)
    elif backend == "fish_speech":
        from fish_speech_tts import generate_audio_fish
        audio, sample_rate = generate_audio_fish(text)
        final_path = create_audio_file(audio, sample_rate=sample_rate, video_id=video_id, video_title=video_title)
    elif backend == "voxtral":
        from voxtral_tts import generate_audio as generate_audio_voxtral
        audio = generate_audio_voxtral(text, voice=voice if voice else None)
        final_path = create_audio_file(audio, video_id=video_id, video_title=video_title)

    elif backend == "vibevoice":
        from vibevoice_tts import generate_audio as generate_audio_vibevoice
        audio, sample_rate = generate_audio_vibevoice(text, voice=voice if voice else "")
        final_path = create_audio_file(audio, sample_rate=sample_rate, video_id=video_id, video_title=video_title)

    elif backend == "phonon":
        from phonon_tts import generate_audio_phonon
        audio, sample_rate = generate_audio_phonon(text)
        final_path = create_audio_file(audio, sample_rate=sample_rate, video_id=video_id, video_title=video_title)

    else:
        if backend != "kokoro":
            print(f"[WARNING] [{_ts()}] [TTS subprocess] Unknown TTS_BACKEND={backend!r}, falling back to kokoro")
        from kokoro_tts import generate_audio
        audio = generate_audio(text)
        final_path = create_audio_file(audio, video_id=video_id, video_title=video_title)
    print(f"[INFO]    [{_ts()}] [TTS subprocess] Audio saved: {final_path}")
    return final_path

def translate_worker(video_id: str, raw_hindi_text: str) -> str:
    """Translate Hindi text to English in an isolated subprocess."""
    started = time.perf_counter()
    try:
        print(
            f"[INFO]    [{_ts()}] [TRANSLATE subprocess] Starting, "
            f"video_id={video_id}, words={len(raw_hindi_text.split())}"
        )
        from youtube_transcript_fetcher import _translate_hindi_to_english
        result = _translate_hindi_to_english(raw_hindi_text)
        print(f"[INFO]    [{_ts()}] [TRANSLATE subprocess] Complete, len={len(result)} chars")
        return result
    finally:
        elapsed = time.perf_counter() - started
        print(
            f"[TIME]    [{_ts()}] [TRANSLATE subprocess] "
            f"Translation took: {format_elapsed(elapsed)}"
        )
