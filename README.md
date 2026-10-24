# Podcast Summarizer

Local Flask app that summarizes YouTube transcripts, generates spoken audio, and supports follow-up questions.

## Local Setup (macOS)

1. Install Python 3.11 or newer, Git, and [GitHub Copilot CLI](https://docs.github.com/en/copilot/how-tos/copilot-cli/set-up-copilot-cli/install-copilot-cli). Sign in with `copilot login`.

2. Install uv. Install FFmpeg only if you plan to use the YouTube Audio Queue:
   ```sh
   brew install uv
   ```
   ```sh
   brew install ffmpeg
   ```

3. Create the virtual environment and install the project:
   ```sh
   uv venv --python 3.11
   source .venv/bin/activate
   uv sync
   ```

4. The repository `.env` is configured for Copilot CLI and Kokoro. Its relevant settings are:
   ```dotenv
   CONDENSATION_ADAPTER=copilot_cli
   COPILOT_MODEL=gpt-6-luna
   COPILOT_CONTEXT=long_context
   ASR_BACKEND=whisper
   TTS_BACKEND=kokoro
   AUTO_SEND_TELEGRAM=false
   ```

5. Start the app:
   ```sh
   python app.py
   ```
   Open <http://localhost:5000>.

For the normal Transcript Queue, YouTube captions are used directly; no ASR model is needed when English captions are available. If only Hindi captions are available, the fallback downloads `sarvamai/sarvam-translate` to translate them.

Kokoro TTS uses the cached `hexgrad/Kokoro-82M` model and configured voice (`af_sarah` by default). Its worker forces Hugging Face offline mode, so audio generation makes no Hugging Face requests; if a required cached file is missing, it fails locally instead of downloading it. Kokoro runs on the CPU. The optional Audio Queue uses Whisper (`mlx-community/whisper-large-v3-mlx`) and requires FFmpeg. LM Studio is not required for the YouTube-to-audio flow. The separate follow-up chat feature still uses LM Studio. Email and Telegram credentials are only needed for those delivery options.

SciPy is pinned to a macOS-compatible release because Kokoro imports Transformers components that load SciPy's compiled sparse extensions.

## TTS Backend

Kokoro remains the default. To switch to Phonon, set this in `.env` and restart Flask:

```dotenv
TTS_BACKEND=phonon
```

After adding the Phonon support, run `uv sync` once to install the pinned `ptts` Python package. The package includes a prebuilt Apple-silicon wheel with Metal support; no separate Rust toolchain or Homebrew CLI install is needed for this CPython setup. Phonon's Rust runtime does not load PyTorch; the existing project still keeps PyTorch installed for Kokoro and the other backends.

Phonon downloads its default English Phonon 31m model the first time it generates audio, then reuses the local cache. An internet connection is needed for that first generation; later runs can use the cached model offline. Model files are not downloaded by `uv sync`.

Optional settings:

```dotenv
PHONON_LANG=en
# PHONON_MODEL_ID=hf://gradium/phonon-31m-en
# PHONON_VOICE=Zoey
PHONON_VOICE_POOL=Zoey,Juniper,Harper,Delia
# PHONON_VOICE_FILE=voices/myvoice.safetensors
```

Leave `PHONON_MODEL_ID` unset to use `ptts`'s pinned default Phonon 31m checkpoint. It can be overridden with a local checkpoint path or Hugging Face model ID. Named Phonon voices come from a separate catalog and are downloaded on first use. With `PHONON_VOICE` unset the voice is chosen by the rotation described below. Kokoro voice names such as `af_sarah` are not compatible with Phonon.

#### Voice rotation

By default Phonon picks one voice at random for every audio generation, so the output does not sound the same every time. The voices come from `PHONON_VOICE_POOL` in `.env`, for example `PHONON_VOICE_POOL=Zoey,Juniper,Harper,Delia`. No voice names live in the code, so to add a voice later you only append its name from the Phonon voice catalog to that line and restart the app (separate names with commas; names with spaces such as `Master Aldric` work, matching ignores case and accents, and unknown names are skipped with a warning). If the pool is empty the model's default voice is used. A voice is never used twice in a row, and a video keeps the same voice when you retry or resume it.

The voice is chosen in this order:

1. `PHONON_VOICE_FILE` (your cloned voice) always wins and turns rotation off.
2. `PHONON_VOICE` pins one voice and turns rotation off. Leave it empty or unset to rotate.
3. `PHONON_VOICE_POOL` rotation.
4. Otherwise the model's default voice.

Each voice other than the bundled one is downloaded the first time it is used (a few hundred KB each). If a voice cannot be downloaded, for example behind a proxy that blocks Hugging Face, it is skipped with a warning and another voice is used instead, so audio generation never fails because of it. Live Speech shows the voice in use, for example `voice Harper`.

#### Custom cloned voice (one-time clone, offline afterwards)

Clone a voice once from about 10 seconds of clean speech (with the speaker's consent), then reuse the resulting `.safetensors` file offline with no reference audio:

```bash
.venv/bin/python scripts/clone_voice.py assets/reference.wav --lang en -o voices/myvoice.safetensors
```

This needs `GRADIUM_API_KEY` in `.env` and network access to `api.gradium.ai` (the only online step). Then set `PHONON_VOICE_FILE=voices/myvoice.safetensors`; it takes precedence over `PHONON_VOICE`. Keep a backup of the file: `voices/` is gitignored. The free Gradium plan allows 5 custom voices and excludes commercial use.

The Phonon 31m weights are Apache-2.0. Phonon 40m checkpoints are gated and licensed CC BY-NC 4.0, so they are not the default. On this Apple-silicon setup, the native runtime selects its available device automatically. The app keeps the same subprocess, WAV output, and delivery flow when the backend changes.

### Live Speech (real-time playback)

The right side of the page has a **Live Speech** workspace: paste text on the left, press **Play live**, and the audio plays while it is being generated. Text of any length works because it is split into sentences and streamed one at a time (`POST /stream_audio`, raw 16-bit mono PCM played with the Web Audio API). On an M4 Pro Phonon generates about 23x faster than real time.

- **Phonon only.** Live playback is enabled only when `TTS_BACKEND=phonon`. With Kokoro (or any other backend) the page shows a notice explaining how to switch, Play is disabled, and `/stream_audio` returns an error.
- **Pause / Resume** keeps your place; **Stop** resets, and the next Play starts from the beginning.
- **Speed** (0.8x to 2x) can be changed while listening and is remembered in the browser. It uses a pitch-preserving time stretch (WSOLA, `static/speech_stretcher.js`), so the voice keeps its natural pitch at every speed.
- **Save as audio file** stays available for every backend and uses the normal file pipeline.
- The Phonon model stays loaded in the Flask process after the first Play, so the first Play after a restart takes a few extra seconds. One stream plays at a time (a second one, for example from another tab, gets a clear "still playing" message after 3 seconds instead of hanging), and it uses the `PHONON_*` settings above, including `PHONON_VOICE_FILE` for your cloned voice (loaded once with the resident model, so restart Flask after changing it).
- Quick Email sits below Live Speech, and the chat panel is on the bottom row.

Copilot CLI is used for transcript condensation and intro generation; its tools are disabled. Context, token budgets, and chunking are configurable through `COPILOT_*` and `CONDENSATION_*` environment variables.

Automatic Telegram delivery is disabled by default. Set `AUTO_SEND_TELEGRAM=true` only when you want the auto-processor to send generated audio and summaries to Telegram.

Copilot logs include request number, stage, prompt size, elapsed time, and a heartbeat every 15 seconds by default (`COPILOT_PROGRESS_INTERVAL_SECONDS`). With `COPILOT_DEBUG=true`, logs also include the exact command and full prompt sent on stdin. This can include the complete transcript; turn it off after debugging. On empty output or CLI failure, bounded stderr is logged with common token formats redacted.

For CLI-internal diagnostics, set `COPILOT_CLI_LOG_LEVEL=debug` and restart Flask. Each invocation prints its native log directory; by default this is a private temporary directory outside the repository. Optionally set `COPILOT_CLI_LOG_DIR` to an explicit directory. These native logs are not automatically redacted and may contain sensitive information; review and redact them before sharing. Unset `COPILOT_CLI_LOG_LEVEL` to disable the extra flags. This setting is separate from `COPILOT_DEBUG`, which prints full input prompts. Heartbeats indicate only that the CLI process is pending, not that inference has started. The application's `COPILOT_TIMEOUT_SECONDS` does not control internal CLI catalog-request timeouts.

## Condensation Limits (Current Defaults)

If standalone Copilot works but the app times out loading the model catalog, compare the logged executable and environment-variable names with the working terminal. Pin that executable with `COPILOT_CLI_EXECUTABLE=/absolute/path/to/copilot` if needed. The child inherits the Flask process's login/token, proxy, and CA environment; shell aliases and later terminal changes do not apply to an already-running Flask process. Restart Flask from the working terminal after changing its environment. Catalog timeouts happen before inference, not because of transcript size.

If the working terminal uses `noproxy`, set `COPILOT_USE_PROXY=false` in `.env` (enabled in this local setup). This removes HTTP/HTTPS/ALL proxy variables, in both cases, only for the Copilot subprocess. Other app requests, credentials, CA configuration, and TLS verification are unchanged. Default: `true` to inherit proxies.

Processing order and token limits:

1. Count tokens with `o200k_base`; the configured Copilot `long_context` window is budgeted as 1,000,000 tokens.
2. Use one direct-summary call when input is within the 100,000-token setting and fits the context/output budget. With the current 15,000-token output budget, 1,024-token safety reserve, and 0.4 MAP output ratio, the effective input ceiling is about 34,940 tokens before prompt overhead.
3. Otherwise, split for MAP at up to 100,000 tokens per chunk, with 2,000-token overlap. The same output/context budget applies, so the current effective chunk size is also about 34,940 tokens before prompt overhead. MAP makes one Copilot call per chunk.
4. REDUCE processes batches sequentially, combining at most 3 mapped chunks per call and preserving up to 512 tokens of previous-section context. Its input budget is at most 13,976 tokens before prompt overhead.
5. Recursively consolidate when combined output exceeds 40,000 tokens or cannot fit the REDUCE budget; with current defaults, the smaller REDUCE budget is the practical trigger.
6. Generate the intro from up to 3 samples, capped at 8,000 input tokens. One-pass processing uses excerpts from the final script.

One-pass processing normally uses 2 Copilot calls: summary and intro (the intro is skipped if it cannot fit). MAP/REDUCE uses one call per MAP chunk, one per sequential REDUCE batch, any required consolidation calls, and an intro call. Each CLI call times out after 600 seconds; the checkpoint permits up to 30 failed attempts per step. Copilot CLI has no hard output-token flag here: the 15,000-token output budget is checked after generation.

If `python app.py` reports that `langchain_community` is missing, sync the updated project dependencies with `uv sync` before starting the app again.
