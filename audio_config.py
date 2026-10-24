import os

ASR_BACKEND = os.getenv("ASR_BACKEND", "whisper")
TTS_BACKEND = os.getenv("TTS_BACKEND", "kokoro")
QWEN_OMNI_MODEL_ID = os.getenv("QWEN_OMNI_MODEL_ID", "Qwen/Qwen2.5-Omni-3B")
QWEN_OMNI_SPEAKER = os.getenv("QWEN_OMNI_SPEAKER", "Chelsie")
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_sarah")
KOKORO_LANG_CODE = os.getenv("KOKORO_LANG_CODE", "a")
PHONON_MODEL_ID = os.getenv("PHONON_MODEL_ID", "")
PHONON_LANG = os.getenv("PHONON_LANG", "en")
PHONON_VOICE = os.getenv("PHONON_VOICE", "")
PHONON_VOICE_FILE = os.getenv("PHONON_VOICE_FILE", "")
PHONON_VOICE_POOL = os.getenv("PHONON_VOICE_POOL", "Zoey,Juniper,Harper,Delia")

def _parse_kokoro_speed() -> float:
    raw = os.getenv("KOKORO_SPEED", "1.0")
    try:
        val = float(raw)
    except ValueError:
        print(f"[WARNING] KOKORO_SPEED='{raw}' is not a valid float — falling back to 1.0")
        return 1.0
    if val < 0.5 or val > 2.0:
        clamped = max(0.5, min(2.0, val))
        print(f"[WARNING] KOKORO_SPEED={val} is outside the safe range [0.5, 2.0] — clamping to {clamped}")
        return clamped
    return val

KOKORO_SPEED: float = _parse_kokoro_speed()
WHISPER_MODEL_ID = os.getenv("WHISPER_MODEL_ID", "mlx-community/whisper-large-v3-mlx")
FISH_SPEECH_CHECKPOINT_PATH = os.getenv("FISH_SPEECH_CHECKPOINT_PATH", "checkpoints/fish-speech-1.5")
FISH_SPEECH_DECODER_CONFIG = os.getenv("FISH_SPEECH_DECODER_CONFIG", "firefly_gan_vq")
FISH_SPEECH_DECODER_CHECKPOINT = os.getenv("FISH_SPEECH_DECODER_CHECKPOINT", "checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth")
FISH_SPEECH_DEVICE = os.getenv("FISH_SPEECH_DEVICE", "")
FISH_SPEECH_PRECISION = os.getenv("FISH_SPEECH_PRECISION", "bfloat16")
FISH_SPEECH_REF_AUDIO = os.getenv("FISH_SPEECH_REF_AUDIO", "")
VOXTRAL_MODEL_ID = os.getenv("VOXTRAL_MODEL_ID", "mlx-community/Voxtral-4B-TTS-2603-mlx-6bit")

# ── VibeVoice TTS backend ────────────────────────────────────────────────────
# Model:  microsoft/VibeVoice-Realtime-0.5B
# Voices: .pt KV-cache tensors shipped with the cloned VibeVoice repo
# Requires: pip install -e "./VibeVoice[streamingtts]"
#
# VIBEVOICE_MODEL_PATH  — local HuggingFace snapshot directory
# VIBEVOICE_VOICES_DIR  — directory containing <voice>.pt preset files
# VIBEVOICE_VOICE       — default voice name (stem of a .pt file or built-in)
# VIBEVOICE_DDPM_STEPS  — DDPM inference steps; higher = slower but smoother
#                         recommended range: 20–50 (default 30)
VIBEVOICE_MODEL_PATH = os.getenv("VIBEVOICE_MODEL_PATH", "./models/VibeVoice-Realtime-0.5B")
VIBEVOICE_VOICES_DIR = os.getenv("VIBEVOICE_VOICES_DIR", "./VibeVoice/demo/voices/streaming_model")
VIBEVOICE_VOICE = os.getenv("VIBEVOICE_VOICE", "en-Davis_man")

def _parse_vibevoice_ddpm_steps() -> int:
    raw = os.getenv("VIBEVOICE_DDPM_STEPS", "30")
    try:
        return max(1, int(float(raw)))
    except ValueError:
        print(f"[WARNING] VIBEVOICE_DDPM_STEPS='{raw}' is not a valid integer — falling back to 30")
        return 30

VIBEVOICE_DDPM_STEPS: int = _parse_vibevoice_ddpm_steps()
