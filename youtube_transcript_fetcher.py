"""YouTube transcript fetching.

Two distinct paths — callers choose which one they need:

  get_youtube_transcript(url)
      Transcript-queue path.  Two-step fallback chain, no audio download:
        1. fetch(video_id, languages=['en']) — fast English path.
        2. fetch(video_id, languages=['hi']) + Sarvam-Translate 4B — loads the
           model locally, translates Hindi → English in chunks, then unloads
           the model to free RAM.
      Returns an "Error:" string when both steps fail.

  get_transcript_via_whisper(url, video_id)   [from whisper_transcriber]
      Audio-queue path.  Always downloads audio via yt-dlp + mlx-whisper.
"""

import gc
import hashlib
import json
import os
import platform
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
from youtube_transcript_api import YouTubeTranscriptApi
from process_runner import run_in_subprocess
import model_worker

# get_transcript_via_whisper is NOT used here — it belongs to the audio-queue
# path only. Import it from whisper_transcriber directly when needed.

_IS_APPLE_SILICON: bool = (
    platform.processor() == "arm" and platform.system() == "Darwin"
)

_SARVAM_MODEL_ID: str = "sarvamai/sarvam-translate"

# Max words per translation chunk — keeps prompt + output within the model's
# 8k-token context window.  ~1500 words ≈ 2000 tokens input; output is
# similarly sized, leaving comfortable headroom.
_TRANSLATE_CHUNK_WORDS: int = 1500

TRANSLATION_CACHE_DIR: Path = Path("translation_cache")
TRANSLATION_CACHE_TTL_DAYS: int = 7


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _translation_cache_key(video_id: str) -> str:
    canonical = f"{video_id}|{_SARVAM_MODEL_ID}"
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _load_translation_cache(key: str) -> str | None:
    cache_path = TRANSLATION_CACHE_DIR / f"{key}.json"
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    try:
        expires_at = datetime.fromisoformat(data["expires_at"])
        if datetime.now(timezone.utc) >= expires_at:
            print(f"[INFO]    [{_ts()}] Translation cache expired for key {key}, ignoring.")
            return None
    except Exception:
        return None
    return data["translated_text"]


def _save_translation_cache(key: str, text: str) -> None:
    try:
        TRANSLATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_key": key,
            "model_id": _SARVAM_MODEL_ID,
            "translated_text": text,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": (datetime.now(timezone.utc) + timedelta(days=TRANSLATION_CACHE_TTL_DAYS)).isoformat(),
        }
        final_path = TRANSLATION_CACHE_DIR / f"{key}.json"
        tmp_path = TRANSLATION_CACHE_DIR / f"{key}.tmp"
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_path, final_path)
        print(f"[INFO]    [{_ts()}] Translation cached to {final_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save translation cache for {key}: {e}")
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def purge_expired_translation_cache() -> int:
    if not TRANSLATION_CACHE_DIR.exists():
        return 0
    count = 0
    for tmp_file in TRANSLATION_CACHE_DIR.glob("*.tmp"):
        try:
            tmp_file.unlink()
            count += 1
        except Exception:
            pass
    for json_file in TRANSLATION_CACHE_DIR.glob("*.json"):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            expires_at = datetime.fromisoformat(data["expires_at"])
            if datetime.now(timezone.utc) >= expires_at:
                json_file.unlink()
                count += 1
        except Exception:
            try:
                json_file.unlink()
                count += 1
            except Exception:
                pass
    if count > 0:
        print(f"[INFO]    [{_ts()}] Purged {count} expired translation cache entries.")
    return count


def _translate_hindi_to_english(hindi_text: str) -> str:
    """Translate Hindi text to English using Sarvam-Translate (4B, local).

    Loads the model, translates in chunks to stay within the 8k-token context
    window, then explicitly unloads the model and frees RAM.
    """
    # Lazy imports — torch and transformers are heavy; only pay the cost when
    # a Hindi transcript is actually encountered.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not hindi_text.strip():
        raise ValueError("Empty Hindi transcript — nothing to translate")

    words = hindi_text.split()
    chunks = [
        " ".join(words[i : i + _TRANSLATE_CHUNK_WORDS])
        for i in range(0, len(words), _TRANSLATE_CHUNK_WORDS)
    ]

    print(f"[INFO]    [{_ts()}] Loading {_SARVAM_MODEL_ID} for Hindi → English translation...")
    # model and tokenizer are initialised to None so the finally block can
    # safely del them even if from_pretrained() raises before assignment.
    model = None
    tokenizer = None
    try:
        if _IS_APPLE_SILICON:
            model = AutoModelForCausalLM.from_pretrained(
                _SARVAM_MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": "mps"}
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                _SARVAM_MODEL_ID, torch_dtype="auto", device_map="auto"
            )
        tokenizer = AutoTokenizer.from_pretrained(_SARVAM_MODEL_ID)

        translated_chunks: list[str] = []
        for i, chunk in enumerate(chunks):
            print(
                f"[INFO]    [{_ts()}] Translating chunk {i + 1}/{len(chunks)} "
                f"({len(chunk.split()):,} words)..."
            )
            messages = [
                {"role": "system", "content": "Translate the text below to English."},
                {"role": "user", "content": chunk},
            ]
            text_input = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=2048,
                do_sample=True,
                temperature=0.01,
                num_return_sequences=1,
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
            output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            # Explicitly free GPU tensors now so Metal/CUDA memory is returned
            # to the pool before the next chunk — mirrors qwen_omni_backend pattern.
            del model_inputs, generated_ids
            translated_chunks.append(output_text.strip())

        return " ".join(translated_chunks)
    finally:
        del model, tokenizer
        gc.collect()
        if _IS_APPLE_SILICON:
            torch.mps.empty_cache()
        else:
            torch.cuda.empty_cache()
        print(f"[INFO]    [{_ts()}] Sarvam-Translate unloaded, RAM freed.")


def extract_video_id(video_link: str) -> str | None:
    """Extract the 11-character video ID from any recognised YouTube URL format.

    Handles: youtube.com/watch, youtu.be, /shorts/, /embed/, /live/,
    and youtube.com/?v= query-param variants.
    Returns None if no valid ID can be found.
    """
    pattern = (
        r"(?:youtube\.com/(?:[^/\n\s]+/\S+/|(?:v|e(?:mbed)?|live)/|\S*?[?&]v=)"
        r"|youtu\.be/|youtube\.com/shorts/)"
        r"([a-zA-Z0-9_-]{11})"
    )
    match = re.search(pattern, video_link)
    return match.group(1) if match else None


def _fetch_transcript_text(video_id: str, language: str) -> str:
    """Fetch captions directly without changing the process proxy environment."""
    with requests.Session() as session:
        session.trust_env = False
        session.verify = (
            os.getenv("REQUESTS_CA_BUNDLE")
            or os.getenv("CURL_CA_BUNDLE")
            or True
        )
        fetched = YouTubeTranscriptApi(http_client=session).fetch(
            video_id, languages=[language]
        )
        return " ".join(entry.text for entry in fetched)


def get_youtube_transcript(video_link: str) -> str:
    """Fetch an English transcript for a YouTube video using the Transcript API only.

    This is the **transcript-queue path**.  It never downloads audio and never
    calls Whisper.

    Two-step fallback chain:
      1. fetch(video_id, languages=['en']) — fast path for English tracks.
      2. fetch(video_id, languages=['hi']) + Sarvam-Translate 4B (local model).
         Loads the model, translates Hindi → English in chunks, then unloads.
    Returns an ``"Error:"`` string when both steps fail.

    Returns:
        Plain English transcript text on success.
        A string prefixed with ``"Invalid YouTube"`` or ``"Error:"`` on failure.
    """
    video_id = extract_video_id(video_link)
    if not video_id:
        return "Invalid YouTube URL format."

    print(f"[INFO]    [{_ts()}] Fetching transcript via API for video: {video_id}")

    # Step 1 — try English directly.
    try:
        text = _fetch_transcript_text(video_id, "en")
        print(f"[INFO]    [{_ts()}] English transcript fetched: {len(text):,} chars")
        return text
    except Exception:
        print(f"[INFO]    [{_ts()}] No English transcript for {video_id}, trying Hindi + Sarvam-Translate...")

    # Step 2 — fetch Hindi transcript and translate locally via Sarvam-Translate.
    # Check translation cache first — a cache hit skips the 4B model load entirely.
    # Cache infrastructure errors are isolated so they never mask transcript errors.
    cache_key: str | None = None
    try:
        cache_key = _translation_cache_key(video_id)
        cached = _load_translation_cache(cache_key)
        if cached is not None:
            print(f"[INFO]    [{_ts()}] Translation cache hit for {video_id}: {len(cached):,} chars")
            return cached
    except Exception as e:
        print(f"[WARNING] Translation cache lookup failed for {video_id}: {e}, proceeding without cache.")

    try:
        hindi_text = _fetch_transcript_text(video_id, "hi")
        print(f"[INFO]    [{_ts()}] Hindi transcript fetched: {len(hindi_text):,} chars, translating...")
        english_text = run_in_subprocess(model_worker.translate_worker, video_id, hindi_text)
        print(f"[INFO]    [{_ts()}] Translation complete: {len(english_text):,} chars")
        if english_text and cache_key is not None:
            _save_translation_cache(cache_key, english_text)
        return english_text
    except Exception as e:
        print(f"[WARNING] No accessible Hindi transcript for {video_id}: {e}")
        return f"Error: No accessible English or Hindi transcript for video {video_id} ({e})."