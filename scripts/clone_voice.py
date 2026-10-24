"""One-time Phonon voice cloning via the Gradium API.

Uploads ~10 s of clean speech and saves the resulting voice embedding as a
.safetensors file that Phonon can load offline.  Adapted from
gradium-ai/xn-ptts scripts/clone-voice.py (MIT OR Apache-2.0).

    python scripts/clone_voice.py assets/reference.wav --lang en -o voices/myvoice.safetensors

Needs GRADIUM_API_KEY in the environment or .env. Clone only voices you have
the speaker's consent to use.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

API = "https://api.gradium.ai/api"
MODEL = "default"  # every Phonon checkpoint shares the same voice files
TIMEOUT = 60


def _check(r: requests.Response, *args, **kwargs) -> None:
    if not r.ok:
        try:
            detail = r.json()["detail"]
        except (ValueError, KeyError, TypeError):
            detail = r.text[:300]
        sys.exit(f"[ERROR]   Gradium API error {r.status_code}: {detail}")


def clone_voice(audio: Path, lang: str, name: str, start: float, out: Path) -> Path:
    key = os.environ.get("GRADIUM_API_KEY")
    if not key:
        sys.exit("[ERROR]   GRADIUM_API_KEY is not set")

    session = requests.Session()
    session.headers["x-api-key"] = key
    session.hooks["response"] = _check

    with audio.open("rb") as f:
        r = session.post(
            f"{API}/voices/",
            files={"audio_file": (audio.name, f)},
            data={"name": name, "language": lang, "start_s": start},
            timeout=TIMEOUT,
        )
    voice_id = r.json().get("uid")
    if not voice_id:
        sys.exit(f"[ERROR]   Cloning failed: {r.json().get('error')}")
    print(f"[INFO]    Created voice {voice_id}")

    for _ in range(60):
        found = session.get(
            f"{API}/voices/embeddings/{voice_id}/{MODEL}", timeout=TIMEOUT
        ).json().get("found")
        if found:
            break
        time.sleep(2)
    else:
        sys.exit(f"[ERROR]   Voice {voice_id} not ready yet; retry the download later")

    blob = session.get(
        f"{API}/voices/embeddings/{voice_id}/{MODEL}/blob",
        params={"as_safetensors": "1"},
        timeout=TIMEOUT,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob.content)
    print(f"[INFO]    Wrote {out} (voice id {voice_id})")
    return out


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("audio", type=Path)
    p.add_argument("--lang", required=True, choices=["en", "fr", "de", "es", "pt"])
    p.add_argument("--name", help="voice name in your Gradium account (default: file stem)")
    p.add_argument("--start", type=float, default=0.0, help="speech start offset in seconds")
    p.add_argument("-o", "--out", type=Path, default=Path("voices/myvoice.safetensors"))
    args = p.parse_args()
    clone_voice(args.audio, args.lang, args.name or args.audio.stem, args.start, args.out)


if __name__ == "__main__":
    main()
