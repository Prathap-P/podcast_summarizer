"""Voice selection for the Phonon backend.

Priority, highest first:
1. PHONON_VOICE_FILE  - a cloned voice file; rotation is off.
2. PHONON_VOICE       - a single pinned catalog voice; rotation is off.
3. PHONON_VOICE_POOL  - pick one voice at random per generation, never the
                        same voice twice in a row.
4. Nothing configured - the model's own default voice.
"""

import random
import re
import threading
from datetime import datetime

_lock = threading.Lock()
_last_voice: str | None = None
_unavailable: set[str] = set()
_catalog_cache: dict[str, str] | None = None


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _catalog_names() -> dict[str, str] | None:
    """Map lower-case catalog voice names to their canonical spelling."""
    global _catalog_cache
    if _catalog_cache is None:
        try:
            import ptts

            _catalog_cache = {
                str(v["name"]).lower(): str(v["name"])
                for v in ptts.catalog_voices()
                if v.get("name")
            }
        except Exception as e:
            print(
                f"[WARNING] [{_ts()}] [PHONON_VOICES] Could not read the voice "
                f"catalog ({e}); voice names are not validated"
            )
            return None
    return _catalog_cache


def parse_voice_pool(raw: str) -> list[str]:
    """Turn 'Zoey, Juniper Harper' into a de-duplicated, validated name list."""
    names: list[str] = []
    seen: set[str] = set()
    catalog = _catalog_names()
    for token in re.split(r"[,\s;]+", raw or ""):
        token = token.strip()
        if not token or token.lower() in seen:
            continue
        seen.add(token.lower())
        if catalog is not None:
            canonical = catalog.get(token.lower())
            if canonical is None:
                print(
                    f"[WARNING] [{_ts()}] [PHONON_VOICES] Unknown voice "
                    f"{token!r} in PHONON_VOICE_POOL - skipping it"
                )
                continue
            token = canonical
        names.append(token)
    return names


def _settings() -> tuple[str, str, str]:
    from audio_config import PHONON_VOICE, PHONON_VOICE_FILE, PHONON_VOICE_POOL

    return (
        PHONON_VOICE_FILE.strip(),
        PHONON_VOICE.strip(),
        PHONON_VOICE_POOL,
    )


def pool() -> list[str]:
    """The configured rotation pool (may be empty)."""
    return parse_voice_pool(_settings()[2])


def mode() -> str:
    """Return 'cloned', 'pinned', 'rotating' or 'default'."""
    voice_file, pinned, raw_pool = _settings()
    if voice_file:
        return "cloned"
    if pinned:
        return "pinned"
    return "rotating" if parse_voice_pool(raw_pool) else "default"


def describe() -> str:
    """One plain-English line describing which voice(s) will be used."""
    voice_file, pinned, raw_pool = _settings()
    if voice_file:
        return "Voice: your cloned voice file (rotation is off)."
    if pinned:
        return (
            f"Voice: {pinned} (pinned by PHONON_VOICE). "
            "Clear PHONON_VOICE to rotate between voices."
        )
    names = parse_voice_pool(raw_pool)
    if names:
        return f"Voices rotate randomly: {', '.join(names)}."
    return "Voice: the model's default voice."


def pick_voice(
    names: list[str],
    previous: str | None = None,
    rng: random.Random | None = None,
) -> str:
    """Random choice that avoids repeating the previous voice when it can."""
    if not names:
        return ""
    candidates = [n for n in names if n != previous] or list(names)
    return (rng or random).choice(candidates)


def select_voice() -> str:
    """Choose the voice for ONE audio generation.

    Returns '' when a cloned file, the model default, or a pinned voice should
    be used instead (those are resolved by the generator itself).
    """
    global _last_voice
    if mode() != "rotating":
        return ""
    names = [n for n in pool() if n not in _unavailable]
    if not names:
        print(
            f"[WARNING] [{_ts()}] [PHONON_VOICES] No usable voice left in "
            "PHONON_VOICE_POOL - using the model default"
        )
        return ""
    with _lock:
        chosen = pick_voice(names, _last_voice)
        _last_voice = chosen
    print(
        f"[INFO]    [{_ts()}] [PHONON_VOICES] Voice selected: {chosen} "
        f"(random from {len(names)})"
    )
    return chosen


def fallback_order(chosen: str) -> list[str | None]:
    """Voices to try, in order, when `chosen` cannot be loaded.

    None at the end means the model's default voice.
    """
    order: list[str | None] = [chosen]
    order += [n for n in pool() if n != chosen and n not in _unavailable]
    order.append(None)
    return order


def label_for(resolved: str) -> str:
    """Human-readable name of the voice that is actually being used."""
    if resolved:
        return resolved
    voice_file, pinned, _ = _settings()
    if voice_file:
        return "your cloned voice"
    return pinned or "default voice"


def mark_unavailable(voice: str) -> None:
    """Remember a voice that failed to load so later picks skip it."""
    with _lock:
        _unavailable.add(voice)


def reset_state() -> None:
    """Forget the last pick and unavailable voices (used by tests)."""
    global _last_voice, _catalog_cache
    with _lock:
        _last_voice = None
        _unavailable.clear()
        _catalog_cache = None
