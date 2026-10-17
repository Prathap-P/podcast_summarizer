from datetime import datetime
import json
import os
from typing import Optional, Protocol

import tiktoken

from system_prompts import *
from utils import remove_thinking_tokens
from condensation_cache import save_checkpoint, MAX_RETRIES_PER_STEP

_PROFILE_VERSION = "condensation-v2"
_PROMPT_VERSION = "narrative-map-reduce-v1"
_PROMPT_SCHEMA_VERSION = "final-script-v1"


class CondensationAdapter(Protocol):
    identity: str
    context_window_tokens: int
    max_completion_tokens: int

    def generate(self, prompt: str) -> str: ...


def _setting(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _ratio_setting(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
        return value if 0 < value <= 10 else default
    except (TypeError, ValueError):
        return default


def _reserved_output_budget(adapter: CondensationAdapter) -> int:
    safety_tokens = _setting("CONDENSATION_COMPLETION_SAFETY_TOKENS", 1024)
    return max(0, int(adapter.max_completion_tokens) - safety_tokens)


def _tokenizer():
    return tiktoken.get_encoding(os.getenv("CONDENSATION_TOKENIZER_ENCODING", "o200k_base"))


def _token_count(text: str, encoding=None) -> int:
    return len((encoding or _tokenizer()).encode(text))


def _max_retries() -> int:
    return _setting("CONDENSATION_MAX_RETRIES_PER_STEP", MAX_RETRIES_PER_STEP, 1)


def get_condensation_profile_id(adapter: CondensationAdapter, script_style: str = "summary") -> str:
    """Return a stable cache identity for this adapter and condensation profile."""
    profile = {
        "version": _PROFILE_VERSION,
        "prompt_version": _PROMPT_VERSION,
        "prompt_schema": _PROMPT_SCHEMA_VERSION,
        "style": script_style,
        "adapter": str(adapter.identity),
        "context_window_tokens": int(adapter.context_window_tokens),
        "max_completion_tokens": int(adapter.max_completion_tokens),
        "tokenizer_encoding": os.getenv("CONDENSATION_TOKENIZER_ENCODING", "o200k_base"),
        "one_pass_max_input_tokens": _setting("CONDENSATION_ONE_PASS_MAX_INPUT_TOKENS", 100000),
        "max_input_tokens": _setting("CONDENSATION_MAX_INPUT_TOKENS", 100000, 1),
        "chunk_overlap_tokens": _setting("CONDENSATION_CHUNK_OVERLAP_TOKENS", 2000),
        "reduce_batch_size": _setting("CONDENSATION_REDUCE_BATCH_SIZE", 3, 1),
        "intro_input_tokens": _setting("CONDENSATION_INTRO_MAX_INPUT_TOKENS", 8000, 1),
        "completion_safety_tokens": _setting("CONDENSATION_COMPLETION_SAFETY_TOKENS", 1024),
        "map_output_ratio": _ratio_setting("CONDENSATION_MAP_OUTPUT_RATIO", 0.4),
        "reduce_output_ratio": _ratio_setting("CONDENSATION_REDUCE_OUTPUT_RATIO", 1.0),
        "final_consolidation_threshold_tokens": _setting(
            "CONDENSATION_FINAL_CONSOLIDATION_THRESHOLD_TOKENS", 40000
        ),
        "prompt_reserve_tokens": _setting("CONDENSATION_PROMPT_RESERVE_TOKENS", 256),
        "context_continuity_tokens": _setting("CONDENSATION_CONTEXT_CONTINUITY_TOKENS", 512),
        "max_retries_per_step": _max_retries(),
    }
    return f"{_PROFILE_VERSION}:{json.dumps(profile, sort_keys=True, separators=(',', ':'))}"


def _map_input(text: str, prompts: dict) -> str:
    return f"""
        System:
        {yt_transcript_shortener_system_message}
        Input:
        {prompts['map_prompt'].replace('{chunk_text}', text)}
    """


def _reduce_input(text: str, prompts: dict, previous_context: str = "") -> str:
    template = prompts["reduce_with_context_prompt"] if previous_context else prompts["reduce_prompt"]
    rendered = template.replace("{previous_context}", previous_context)
    rendered = rendered.replace("{combined_map_results}", text)
    return f"""
        System:
        {yt_transcript_shortener_system_message}
        Input:
        {rendered}
    """


def _fits_context(prompt: str, adapter: CondensationAdapter, encoding=None) -> bool:
    return _token_count(prompt, encoding) + int(adapter.max_completion_tokens) <= int(adapter.context_window_tokens)


def _input_budget(
    adapter: CondensationAdapter,
    prompt_builder,
    encoding=None,
    context_tokens: int = 0,
    output_ratio: float = 1.0,
) -> int:
    reserve = _setting("CONDENSATION_PROMPT_RESERVE_TOKENS", 256)
    context_sample = "x " * context_tokens
    overhead = _token_count(prompt_builder("", context_sample), encoding)
    output_budget = int(_reserved_output_budget(adapter) / output_ratio)
    context_budget = (
        int(adapter.context_window_tokens)
        - int(adapter.max_completion_tokens)
        - overhead
        - reserve
    )
    return max(0, min(
        _setting("CONDENSATION_MAX_INPUT_TOKENS", 100000, 1),
        output_budget,
        context_budget,
    ))


def _split_by_tokens(content: str, max_tokens: int, overlap_tokens: int = 0, encoding=None) -> list[str]:
    """Split on character boundaries while enforcing token budgets and preserving Unicode."""
    if not content:
        return []
    if max_tokens < 1:
        raise ValueError("Adapter context leaves no input-token budget for condensation")

    tokenizer = encoding or _tokenizer()
    chunks: list[str] = []
    start = 0
    while start < len(content):
        if _token_count(content[start:start + 1], tokenizer) > max_tokens:
            raise ValueError("A single source character exceeds the configured input-token budget")
        low, high = start + 1, len(content)
        while low < high:
            middle = (low + high + 1) // 2
            if _token_count(content[start:middle], tokenizer) <= max_tokens:
                low = middle
            else:
                high = middle - 1
        end = low
        chunks.append(content[start:end])
        if end == len(content):
            break

        overlap = min(max(0, overlap_tokens), max_tokens - 1)
        if overlap == 0:
            start = end
            continue
        low, high = start + 1, end
        while low < high:
            middle = (low + high) // 2
            if _token_count(content[middle:end], tokenizer) <= overlap:
                high = middle
            else:
                low = middle + 1
        start = max(start + 1, low)
    return chunks


def _context_for(text: str, previous_context: str, prompts: dict, adapter: CondensationAdapter, encoding=None) -> str:
    if not previous_context:
        return ""
    tokenizer = encoding or _tokenizer()
    context_limit = _setting("CONDENSATION_CONTEXT_CONTINUITY_TOKENS", 512)
    context_ids = tokenizer.encode(previous_context)
    context = tokenizer.decode(context_ids[-context_limit:]) if context_limit else ""
    while context:
        if _fits_context(_reduce_input(text, prompts, context), adapter, tokenizer):
            return context
        context_ids = tokenizer.encode(context)
        context = tokenizer.decode(context_ids[max(1, len(context_ids) // 2):])
    return ""


def build_intro_prompt(
    adapter: CondensationAdapter,
    map_samples: list[str],
    condensed_content: str,
    script_style: str = "summary",
) -> str:
    """Build a bounded intro prompt, sampling the final script when MAP is skipped."""
    encoding = _tokenizer()
    prompts = analysis_map_reduce_prompts if script_style == "analysis" else map_reduce_custom_prompts
    template = prompts["intro_prompt"]
    overhead = _token_count(template.format(map_samples=""), encoding)
    reserve = _setting("CONDENSATION_PROMPT_RESERVE_TOKENS", 256)
    budget = min(
        _setting("CONDENSATION_INTRO_MAX_INPUT_TOKENS", 8000, 1),
        int(adapter.context_window_tokens)
        - int(adapter.max_completion_tokens)
        - overhead
        - reserve,
    )
    if budget <= 0:
        return ""

    samples = [sample for sample in map_samples if sample.strip()][: min(3, budget)]
    if not samples and condensed_content.strip():
        content_tokens = encoding.encode(condensed_content)
        count = min(3, len(content_tokens), budget)
        sample_size = max(1, budget // count) if count else 0
        starts = [
            0,
            max(0, (len(content_tokens) - sample_size) // 2),
            max(0, len(content_tokens) - sample_size),
        ][:count]
        samples = [
            encoding.decode(content_tokens[start : start + sample_size])
            for start in starts
        ]

    while samples:
        sample_size = max(1, budget // len(samples))
        bounded_samples = [
            encoding.decode(encoding.encode(sample)[:sample_size])
            for sample in samples
        ]
        prompt = template.format(map_samples="\n\n".join(bounded_samples))
        if _token_count(prompt, encoding) + int(adapter.max_completion_tokens) + reserve <= int(adapter.context_window_tokens):
            return prompt
        longest = max(range(len(bounded_samples)), key=lambda index: len(encoding.encode(bounded_samples[index])))
        tokens = encoding.encode(bounded_samples[longest])
        if len(tokens) <= 1:
            samples.pop(longest)
        else:
            samples[longest] = encoding.decode(tokens[: len(tokens) // 2])
    return ""


def _generate_script(
    adapter: CondensationAdapter,
    prompt: str,
    label: str,
    checkpoint: Optional[dict],
    checkpoint_key: Optional[str],
    retry_bucket: Optional[str],
    retry_key: str,
) -> str:
    has_checkpoint = checkpoint is not None and checkpoint_key is not None
    retry_counts = checkpoint.setdefault(retry_bucket, {}) if has_checkpoint and retry_bucket else None
    retries = retry_counts.get(retry_key, 0) if retry_counts is not None else 0
    if retries >= _max_retries():
        raise ValueError(f"{label} exceeded max retries ({_max_retries()}).")

    def record_failure() -> None:
        if retry_counts is not None:
            retry_counts[retry_key] = retry_counts.get(retry_key, 0) + 1
            save_checkpoint(checkpoint_key, checkpoint)  # type: ignore[arg-type]

    try:
        response = adapter.generate(prompt)
        if not isinstance(response, str):
            raise TypeError("adapter.generate(prompt) must return str")
    except Exception as exc:
        record_failure()
        raise ValueError(f"Model crashed during {label}: {exc}") from exc

    cleaned, success = remove_thinking_tokens(response)
    if not success or not cleaned.strip():
        record_failure()
        raise ValueError(f"Failed to remove thinking tokens from {label}")
    return cleaned


def _save_checkpoint(checkpoint_key: Optional[str], checkpoint: Optional[dict]) -> None:
    if checkpoint_key is not None and checkpoint is not None:
        save_checkpoint(checkpoint_key, checkpoint)


def _recursive_consolidate(
    parts: list[str], adapter: CondensationAdapter, prompts: dict, reduce_budget: int, batch_size: int,
    encoding, checkpoint: Optional[dict], checkpoint_key: Optional[str], force_pass: bool,
) -> str:
    has_checkpoint = checkpoint is not None and checkpoint_key is not None
    current_parts = list(checkpoint.get("consolidation_parts", parts)) if has_checkpoint else list(parts)
    round_index = int(checkpoint.get("consolidation_round", 0)) if has_checkpoint else 0
    first_pass = force_pass

    while first_pass or _token_count("\n\n".join(current_parts), encoding) > reduce_budget:
        if round_index >= 8:
            raise ValueError("Recursive consolidation did not fit the safe reduce token budget")
        expanded: list[str] = []
        for part in current_parts:
            if _token_count(part, encoding) <= reduce_budget:
                expanded.append(part)
            else:
                expanded.extend(_split_by_tokens(part, reduce_budget, 0, encoding))

        round_cache = checkpoint.setdefault("consolidation_round_results", {}) if has_checkpoint else {}
        next_parts: list[str] = []
        cursor = 0
        batch_index = 0
        previous_context = ""
        while cursor < len(expanded):
            group = [expanded[cursor]]
            group_context = _context_for("\n\n---\n\n".join(group), previous_context, prompts, adapter, encoding)
            while cursor + len(group) < len(expanded) and len(group) < batch_size:
                candidate = group + [expanded[cursor + len(group)]]
                candidate_text = "\n\n---\n\n".join(candidate)
                candidate_context = _context_for(candidate_text, previous_context, prompts, adapter, encoding)
                if (
                    _token_count(candidate_text, encoding) > reduce_budget
                    or not _fits_context(
                        _reduce_input(candidate_text, prompts, candidate_context),
                        adapter,
                        encoding,
                    )
                ):
                    break
                group = candidate
                group_context = candidate_context

            cache_key = f"{round_index}:{batch_index}"
            if cache_key in round_cache:
                consolidated = round_cache[cache_key]
            else:
                group_text = "\n\n---\n\n".join(group)
                prompt = _reduce_input(group_text, prompts, group_context)
                if not _fits_context(prompt, adapter, encoding):
                    raise ValueError("A consolidation input exceeds the adapter context window")
                retries = int(checkpoint.get("consolidation_retries", 0)) if has_checkpoint else 0
                if retries >= _max_retries():
                    raise ValueError(f"Final consolidation exceeded max retries ({_max_retries()}).")
                try:
                    consolidated = _generate_script(
                        adapter, prompt, f"final consolidation batch {batch_index + 1}",
                        None, None, None, cache_key,
                    )
                except ValueError:
                    if has_checkpoint:
                        checkpoint["consolidation_retries"] = retries + 1
                        save_checkpoint(checkpoint_key, checkpoint)  # type: ignore[arg-type]
                    raise
                if has_checkpoint:
                    round_cache[cache_key] = consolidated
                    checkpoint["consolidation_parts"] = current_parts
                    checkpoint["consolidation_round"] = round_index
                    save_checkpoint(checkpoint_key, checkpoint)  # type: ignore[arg-type]

            next_parts.append(consolidated)
            previous_context = consolidated
            cursor += len(group)
            batch_index += 1

        before_tokens = _token_count("\n\n".join(current_parts), encoding)
        after_tokens = _token_count("\n\n".join(next_parts), encoding)
        if after_tokens >= before_tokens and after_tokens > reduce_budget:
            raise ValueError("Recursive consolidation output did not shrink below the safe token budget")
        current_parts = next_parts
        round_index += 1
        first_pass = False
        if has_checkpoint:
            checkpoint["consolidation_parts"] = current_parts
            checkpoint["consolidation_round"] = round_index
            save_checkpoint(checkpoint_key, checkpoint)  # type: ignore[arg-type]

    return "\n\n".join(current_parts)


def condense_content(
    content: str,
    current_model: CondensationAdapter,
    checkpoint_key: Optional[str] = None,
    checkpoint: Optional[dict] = None,
    script_style: str = "summary",
) -> str:
    """Run map-reduce condensation, resuming from checkpoint if provided.

    Args:
        content:        Raw text to condense.
        current_model:  Adapter implementing generate(prompt) -> str and exposing identity/context limits.
        checkpoint_key: Key for atomic checkpoint saves.  None = no persistence.
        checkpoint:     Mutable checkpoint dict; updated in-place and saved after
                        every step so a crash loses at most one step's work.
        script_style:   'summary' (default) or 'analysis'.

    Returns:
        Condensed text string.
    """
    _has_checkpoint = checkpoint_key is not None and checkpoint is not None
    encoding = _tokenizer()
    _prompts = analysis_map_reduce_prompts if script_style == "analysis" else map_reduce_custom_prompts
    map_budget = _input_budget(
        current_model,
        lambda text, _context: _map_input(text, _prompts),
        encoding,
        output_ratio=_ratio_setting("CONDENSATION_MAP_OUTPUT_RATIO", 0.4),
    )
    reduce_budget = _input_budget(
        current_model,
        lambda text, context: _reduce_input(text, _prompts, context),
        encoding,
        _setting("CONDENSATION_CONTEXT_CONTINUITY_TOKENS", 512),
        _ratio_setting("CONDENSATION_REDUCE_OUTPUT_RATIO", 1.0),
    )

    if _has_checkpoint and checkpoint.get("final_output") is not None:
        return checkpoint["final_output"]

    one_pass_input_budget = min(
        _setting("CONDENSATION_ONE_PASS_MAX_INPUT_TOKENS", 100000),
        int(
            _reserved_output_budget(current_model)
            / _ratio_setting("CONDENSATION_MAP_OUTPUT_RATIO", 0.4)
        ),
    )
    if _token_count(content, encoding) <= one_pass_input_budget:
        direct_prompt = _map_input(content, _prompts)
        if _fits_context(direct_prompt, current_model, encoding):
            direct_output = _generate_script(
                current_model, direct_prompt, "direct summary", checkpoint,
                checkpoint_key, "map_retry_counts", "direct",
            )
            if _has_checkpoint:
                checkpoint["final_output"] = direct_output
                _save_checkpoint(checkpoint_key, checkpoint)
            return direct_output
    if map_budget < 1 or reduce_budget < 1:
        raise ValueError("Adapter context window is too small for condensation prompts and output reservation")

    def _save() -> None:
        if _has_checkpoint:
            save_checkpoint(checkpoint_key, checkpoint)  # type: ignore[arg-type]

    print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] condense_content: Starting condensation for {len(content)} chars")

    # ------------------------------------------------------------------
    # Stage 1 — split into chunks (idempotent; reuse stored split on resume)
    # ------------------------------------------------------------------
    if _has_checkpoint and checkpoint.get("map_chunks"):
        chunks = checkpoint["map_chunks"]
        if not all(
            _token_count(chunk, encoding) <= map_budget
            and _fits_context(_map_input(chunk, _prompts), current_model, encoding)
            for chunk in chunks
        ):
            chunks = []
            checkpoint["map_results"] = {}
            checkpoint["map_retry_counts"] = {}
            checkpoint["reduce_results"] = {}
            checkpoint["reduce_retry_counts"] = {}
            checkpoint["tts_results"] = {}
            checkpoint["consolidation_result"] = None
            checkpoint.pop("consolidation_parts", None)
            checkpoint.pop("consolidation_round_results", None)
            checkpoint.pop("consolidation_round", None)
            checkpoint["map_chunks"] = None
        print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Resuming: reusing {len(chunks)} stored chunks from checkpoint")
    else:
        chunks = []
    if not chunks:
        chunks = _split_by_tokens(
            content,
            map_budget,
            _setting("CONDENSATION_CHUNK_OVERLAP_TOKENS", 2000),
            encoding,
        )
        if _has_checkpoint:
            checkpoint["map_chunks"] = chunks
            _save()
        print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Content split into {len(chunks)} chunks")

    # ------------------------------------------------------------------
    # Stage 2 — MAP phase
    # ------------------------------------------------------------------
    print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Starting MAP phase ({len(chunks)} chunks)")
    processed_chunks: list[str] = []

    for idx, chunk in enumerate(chunks):
        str_idx = str(idx)

        # Resume: skip already-completed chunks
        if _has_checkpoint and str_idx in checkpoint["map_results"]:
            cached = checkpoint["map_results"][str_idx]
            processed_chunks.append(cached)
            print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Resuming: MAP chunk {idx + 1}/{len(chunks)} already complete, skipping")
            continue

        # Retry cap: fail fast if this chunk has already blown its budget
        if _has_checkpoint:
            retries_used = checkpoint["map_retry_counts"].get(str_idx, 0)
            if retries_used >= _max_retries():
                error_msg = (
                    f"MAP chunk {idx + 1}/{len(chunks)} exceeded max retries "
                    f"({_max_retries()}). Aborting — fix the model response or "
                    f"delete the checkpoint to start fresh."
                )
                print(f"[ERROR] {error_msg}")
                raise ValueError(error_msg)

        print(f"[DEBUG] Processing MAP chunk {idx + 1}/{len(chunks)} ({len(chunk)} chars)")
        map_input = _map_input(chunk, _prompts)
        cleaned = _generate_script(
            current_model,
            map_input,
            f"MAP chunk {idx + 1}/{len(chunks)}",
            checkpoint,
            checkpoint_key,
            "map_retry_counts",
            str_idx,
        )

        # Success — persist before moving on
        if _has_checkpoint:
            checkpoint["map_results"][str_idx] = cleaned
            _save()

        processed_chunks.append(cleaned)
        print(f"[DEBUG] MAP chunk {idx + 1} processed: {len(cleaned)} chars")

    print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] MAP phase complete. {len(processed_chunks)} chunks processed")

    reduce_batch_size = _setting("CONDENSATION_REDUCE_BATCH_SIZE", 3, 1)
    reduce_item_budget = max(1, (reduce_budget - 64) // reduce_batch_size)
    reduce_items: list[str] = []
    for mapped_text in processed_chunks:
        if _token_count(mapped_text, encoding) <= reduce_item_budget:
            reduce_items.append(mapped_text)
        else:
            reduce_items.extend(_split_by_tokens(mapped_text, reduce_item_budget, 0, encoding))
    processed_chunks = reduce_items

    # ------------------------------------------------------------------
    # Stage 3 — REDUCE phase
    # ------------------------------------------------------------------
    print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Starting REDUCE phase")

    if len(processed_chunks) <= reduce_batch_size:
        # ---- Single-batch reduce ----
        print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Single batch mode ({len(processed_chunks)} chunks)")

        # Re-use cached result if it exists (covers single-batch reduce+tts resume)
        tts_results_sb = checkpoint.setdefault("tts_results", {}) if _has_checkpoint else {}
        if _has_checkpoint and tts_results_sb.get("0"):
            # Full resume — both reduce and TTS already done
            final_output = tts_results_sb["0"]
            print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Resuming: single-batch REDUCE+TTS already complete, skipping")
        else:
            # Partial resume: reduce already saved but TTS crashed → skip straight to TTS
            reduce_results_sb = checkpoint.get("reduce_results", {}) if _has_checkpoint else {}
            if _has_checkpoint and reduce_results_sb.get("0"):
                cleaned_reduce = reduce_results_sb["0"]
                print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Resuming: single-batch REDUCE already done, running TTS pass only")
            else:
                # Fresh run — execute reduce
                if _has_checkpoint:
                    retry_count = checkpoint["reduce_retry_counts"].get("0", 0)
                    if retry_count >= _max_retries():
                        raise ValueError(
                            f"Single-batch REDUCE exceeded max retries ({_max_retries()})."
                        )

                combined_chunks = "\n\n---\n\n".join(processed_chunks)
                reduce_input = _reduce_input(combined_chunks, _prompts)
                if not _fits_context(reduce_input, current_model, encoding):
                    raise ValueError("Single-batch REDUCE input exceeds the adapter context window")
                cleaned_reduce = _generate_script(
                    current_model, reduce_input, "single-batch REDUCE", checkpoint,
                    checkpoint_key, "reduce_retry_counts", "0",
                )

                if _has_checkpoint:
                    checkpoint.setdefault("reduce_results", {})["0"] = cleaned_reduce
                    checkpoint["reduce_batches_total"] = 1
                    _save()

            # TTS polish pass — TEMPORARILY DISABLED for output length testing
            # tts_output = _run_tts_pass(
            #     cleaned_reduce, current_model, "0", checkpoint, checkpoint_key, _prompts
            # )
            # if _has_checkpoint:
            #     checkpoint.setdefault("tts_results", {})["0"] = tts_output
            #     _save()

            final_output = cleaned_reduce

    else:
        # ---- Multi-batch reduce ----
        num_batches = (len(processed_chunks) + reduce_batch_size - 1) // reduce_batch_size
        print(
            f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Batch mode: {len(processed_chunks)} chunks "
            f"→ {num_batches} batches (with context continuity)"
        )

        if _has_checkpoint and checkpoint.get("reduce_batches_total") is None:
            checkpoint["reduce_batches_total"] = num_batches
            _save()

        # Seed previous_context from the last already-completed batch so that
        # a resumed run doesn't start batch N with empty context.
        completed_batch_indices = sorted(
            int(k) for k in checkpoint["reduce_results"]
        ) if _has_checkpoint else []
        previous_context = (
            checkpoint["reduce_results"][str(completed_batch_indices[-1])]
            if completed_batch_indices else ""
        )

        batch_results: list[str] = []

        for batch_idx in range(num_batches):
            str_batch = str(batch_idx)

            # Case 1: Full resume — both reduce and TTS already done for this batch
            if _has_checkpoint and str_batch in checkpoint.get("tts_results", {}):
                tts_cached = checkpoint["tts_results"][str_batch]
                batch_results.append(tts_cached)
                # previous_context uses reduce result for LLM continuity (factual/dense)
                previous_context = checkpoint.get("reduce_results", {}).get(str_batch, tts_cached)
                print(
                    f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] "
                    f"Resuming: REDUCE+TTS batch {batch_idx + 1}/{num_batches} already complete, skipping"
                )
                continue

            # Case 2: Partial resume — reduce done but TTS crashed for this batch
            reduce_results_mb = checkpoint.get("reduce_results", {}) if _has_checkpoint else {}
            if _has_checkpoint and str_batch in reduce_results_mb:
                cleaned_batch = reduce_results_mb[str_batch]
                print(
                    f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] "
                    f"Resuming: REDUCE batch {batch_idx + 1}/{num_batches} done, running TTS pass only"
                )
            else:
                # Case 3: Fresh run — execute reduce
                if _has_checkpoint:
                    retries_used = checkpoint["reduce_retry_counts"].get(str_batch, 0)
                    if retries_used >= _max_retries():
                        raise ValueError(
                            f"REDUCE batch {batch_idx + 1}/{num_batches} exceeded max retries "
                            f"({_max_retries()})."
                        )

                start_idx = batch_idx * reduce_batch_size
                end_idx = min((batch_idx + 1) * reduce_batch_size, len(processed_chunks))
                batch_chunks = processed_chunks[start_idx:end_idx]

                print(
                    f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] "
                    f"Processing REDUCE batch {batch_idx + 1}/{num_batches} ({len(batch_chunks)} chunks)..."
                )

                combined_batch = "\n\n---\n\n".join(batch_chunks)

                context_snippet = _context_for(
                    combined_batch, previous_context, _prompts, current_model, encoding
                ) if batch_idx else ""
                reduce_input = _reduce_input(combined_batch, _prompts, context_snippet)
                if not _fits_context(reduce_input, current_model, encoding):
                    raise ValueError(f"REDUCE batch {batch_idx + 1}/{num_batches} exceeds the adapter context window")
                cleaned_batch = _generate_script(
                    current_model,
                    reduce_input,
                    f"REDUCE batch {batch_idx + 1}/{num_batches}",
                    checkpoint,
                    checkpoint_key,
                    "reduce_retry_counts",
                    str_batch,
                )

                if _has_checkpoint:
                    checkpoint.setdefault("reduce_results", {})[str_batch] = cleaned_batch
                    _save()

            # TTS polish pass — TEMPORARILY DISABLED for output length testing
            # tts_batch = _run_tts_pass(
            #     cleaned_batch, current_model, str_batch, checkpoint, checkpoint_key, _prompts
            # )
            # if _has_checkpoint:
            #     checkpoint.setdefault("tts_results", {})[str_batch] = tts_batch
            #     _save()

            batch_results.append(cleaned_batch)
            previous_context = cleaned_batch  # use dense reduce result for LLM context continuity
            print(f"[SUCCESS] REDUCE batch {batch_idx + 1} complete: {len(cleaned_batch)} chars")

        final_output = "\n\n".join(batch_results)
        print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] All REDUCE batches combined: {len(final_output)} chars")

        # Stage 4 — optional final consolidation
        consolidation_threshold = _setting("CONDENSATION_FINAL_CONSOLIDATION_THRESHOLD_TOKENS", 40000)
        if _token_count(final_output, encoding) > consolidation_threshold or _token_count(final_output, encoding) > reduce_budget:
            print(
                f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] "
                f"Final consolidation needed ({_token_count(final_output, encoding)} tokens)"
            )

            # Resume: reuse cached consolidation result
            if _has_checkpoint and checkpoint.get("consolidation_result"):
                final_output = checkpoint["consolidation_result"]
                print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] Resuming: final consolidation already complete, skipping")
            else:
                final_output = _recursive_consolidate(
                    batch_results,
                    current_model,
                    _prompts,
                    reduce_budget,
                    reduce_batch_size,
                    encoding,
                    checkpoint,
                    checkpoint_key,
                    force_pass=True,
                )
                if _has_checkpoint:
                    checkpoint["consolidation_result"] = final_output
                    _save()
                print(f"[SUCCESS] Final consolidation complete: {len(final_output)} chars")
                print(f"[INFO]    [{datetime.now().strftime('%H:%M:%S')}] Final consolidation bypasses TTS pass — output is reduce-style prose")
        else:
            print(
                f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] "
                f"No final consolidation needed ({_token_count(final_output, encoding)} tokens)"
            )

    consolidation_threshold = _setting("CONDENSATION_FINAL_CONSOLIDATION_THRESHOLD_TOKENS", 40000)
    if len(processed_chunks) <= reduce_batch_size and (
        _token_count(final_output, encoding) > consolidation_threshold
        or _token_count(final_output, encoding) > reduce_budget
    ):
        if _has_checkpoint and checkpoint.get("consolidation_result"):
            final_output = checkpoint["consolidation_result"]
        else:
            final_output = _recursive_consolidate(
                [final_output], current_model, _prompts, reduce_budget,
                reduce_batch_size, encoding, checkpoint, checkpoint_key, True,
            )
            if _has_checkpoint:
                checkpoint["consolidation_result"] = final_output
                _save()

    # ------------------------------------------------------------------
    # Stage 5 — persist final output
    # ------------------------------------------------------------------
    if _has_checkpoint:
        checkpoint["final_output"] = final_output
        _save()

    print(f"[INFO] [{datetime.now().strftime('%H:%M:%S')}] REDUCE phase complete: {len(final_output)} chars")
    print(f"[SUCCESS] Condensation complete. Original: {len(content)} -> Final: {len(final_output)} chars")
    return final_output


def _run_tts_pass(
    text: str,
    current_model,
    retry_key: str,
    checkpoint: Optional[dict],
    checkpoint_key: Optional[str],
    prompts_dict: Optional[dict] = None,
) -> str:
    """Run the TTS speech-polish pass on a single reduce batch result.

    Sends the reduce output through the tts_prompt to convert dense narrative
    prose into natural spoken-word script (contractions, breath-test sentences,
    prosody markers). The caller is responsible for persisting the return value
    to checkpoint["tts_results"][retry_key].

    Args:
        text:           Reduce batch output to polish.
        current_model:  LangChain LLM instance.
        retry_key:      String key for checkpoint storage (e.g. "0", "1").
        checkpoint:     Mutable checkpoint dict; updated in-place. None = no persistence.
        checkpoint_key: Key for atomic checkpoint saves. None = no persistence.

    Returns:
        TTS-polished text string.

    Raises:
        ValueError: On model crash or missing <final_script> tags (after persisting retry count).
    """
    _has_checkpoint = checkpoint_key is not None and checkpoint is not None
    _prompts = prompts_dict if prompts_dict is not None else map_reduce_custom_prompts

    def _save() -> None:
        if _has_checkpoint:
            save_checkpoint(checkpoint_key, checkpoint)

    # Retry cap
    if _has_checkpoint:
        tts_retry_counts = checkpoint.setdefault("tts_retry_counts", {})
        retries_used = tts_retry_counts.get(retry_key, 0)
        if retries_used >= _max_retries():
            raise ValueError(
                f"TTS pass for batch {retry_key} exceeded max retries ({_max_retries()}). "
                f"Aborting — fix the model response or delete the checkpoint to start fresh."
            )

    tts_input = f"""
        System:
        {yt_transcript_shortener_system_message}
        Input:
        {_prompts["tts_prompt"].replace("{text_to_refine}", text)}
    """

    print(f"[DEBUG] Running TTS pass for batch {retry_key} ({len(text)} chars)...")

    cleaned = _generate_script(
        current_model, tts_input, f"TTS pass for batch {retry_key}", checkpoint,
        checkpoint_key, "tts_retry_counts", retry_key,
    )

    print(f"[DEBUG] TTS pass for batch {retry_key} cleaned: {len(cleaned)} chars")
    return cleaned


def split_content(
    content: str,
    max_input_tokens: Optional[int] = None,
    overlap_tokens: Optional[int] = None,
):
    """Split content by configured token limits, preserving Unicode characters."""
    chunks = _split_by_tokens(
        content,
        max_input_tokens or _setting("CONDENSATION_MAX_INPUT_TOKENS", 100000, 1),
        _setting("CONDENSATION_CHUNK_OVERLAP_TOKENS", 2000) if overlap_tokens is None else overlap_tokens,
    )
    print(f"[DEBUG] split_content: Created {len(chunks)} token-bounded chunks")
    return chunks