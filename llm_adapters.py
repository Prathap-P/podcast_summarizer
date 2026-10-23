"""Text-only adapters used by the condensation pipeline."""

import hashlib
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import tiktoken


def _validate_output_budget(text: str, budget_tokens: int) -> str:
    encoding = tiktoken.get_encoding(
        os.getenv("CONDENSATION_TOKENIZER_ENCODING", "o200k_base")
    )
    output_tokens = len(encoding.encode(text))
    if output_tokens > budget_tokens:
        raise LLMAdapterError(
            f"LLM response exceeded the configured output budget ({output_tokens} > {budget_tokens} tokens)"
        )
    return text


class LLMAdapterError(RuntimeError):
    """Raised when an LLM adapter cannot produce a text response."""


def _cli_diagnostic_hint(stderr: str) -> str:
    text = " ".join(stderr.lower().split())
    if "model catalog request timed out" in text:
        return "model catalog connectivity timeout (before inference)"
    if any(marker in text for marker in ("certificate verify failed", "self signed certificate", "x509", "tls")):
        return "TLS/certificate verification; check corporate proxy CA trust"
    if any(marker in text for marker in ("not logged in", "unauthorized", "authentication", "401", "403")):
        return "Copilot authentication or access"
    if any(marker in text for marker in ("rate limit", "quota", "429")):
        return "Copilot rate limit or quota"
    if any(marker in text for marker in ("context length", "context window", "too many tokens", "token limit")):
        return "model context limit"
    if any(marker in text for marker in ("unknown model", "model not found", "model is not available")):
        return "model availability or plan access"
    if any(marker in text for marker in ("connection", "network", "proxy", "dns", "timed out")):
        return "network or proxy connectivity"
    return "unclassified CLI diagnostic" if text else "no stderr diagnostic"


def _redact_cli_diagnostic(stderr: str, max_chars: int = 2000) -> str:
    text = stderr.strip()
    text = re.sub(
        r"(?i)(authorization\s*:\s*bearer\s+)[^\s]+",
        r"\1[REDACTED]",
        text,
    )
    text = re.sub(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", text)
    text = re.sub(
        r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b",
        "[REDACTED]",
        text,
    )
    text = re.sub(
        r"(?i)\b(token|password|secret|api[_ -]?key)\b(\s*[:=]\s*)\S+",
        r"\1\2[REDACTED]",
        text,
    )
    if len(text) > max_chars:
        text = text[:max_chars] + "...[truncated]"
    return text


def _copilot_stage(prompt: str) -> str:
    text = prompt.lower()
    if "content summary samples" in text or "content analysis samples" in text:
        return "intro"
    if "data to synthesize" in text or "current batch to synthesize" in text:
        return "reduce"
    if "input segment:" in text:
        return "map/direct summary"
    return "condensation"


def _positive_int(settings: Mapping[str, str], name: str, default: int) -> int:
    raw_value = settings.get(name, str(default))
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(settings: Mapping[str, str], name: str, default: float) -> float:
    raw_value = settings.get(name, str(default))
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return value


class LMStudioAdapter:
    """Wrap the configured LangChain model with the text adapter contract."""

    def __init__(
        self,
        model: Any,
        model_key: str,
        base_url: str,
        context_window_tokens: int,
        max_completion_tokens: int,
    ) -> None:
        self._model = model
        endpoint_fingerprint = hashlib.sha256(base_url.encode("utf-8")).hexdigest()[:16]
        self.identity = (
            f"lm_studio:model={model_key}:endpoint={endpoint_fingerprint}:"
            f"context_window_tokens={context_window_tokens}:"
            f"max_completion_tokens={max_completion_tokens}"
        )
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = max_completion_tokens

    def generate(self, prompt: str) -> str:
        response = self._model.invoke(prompt)
        content = getattr(response, "content", None)
        if not isinstance(content, str):
            raise LLMAdapterError("LM Studio returned a non-text response")
        return _validate_output_budget(content, self.max_completion_tokens)


class CopilotCLIAdapter:
    """Run one tool-restricted, text-only Copilot CLI prompt per call.

    The output-token value is a pipeline budget; Copilot CLI does not expose a
    hard completion-token limit through its prompt-mode command.
    """

    def __init__(
        self,
        executable: str,
        model: str,
        context: str,
        context_window_tokens: int,
        max_completion_tokens: int,
        timeout_seconds: float,
        progress_interval_seconds: float,
        use_proxy: bool = True,
        cli_log_level: str = "",
        cli_log_dir: str = "",
    ) -> None:
        if not executable.strip():
            raise ValueError("COPILOT_CLI_EXECUTABLE must not be empty")
        if not model.strip():
            raise ValueError("COPILOT_MODEL must not be empty")
        if context not in {"default", "long_context"}:
            raise ValueError("COPILOT_CONTEXT must be 'default' or 'long_context'")
        if cli_log_level not in {"", "none", "error", "warning", "info", "debug", "all", "default"}:
            raise ValueError("Invalid COPILOT_CLI_LOG_LEVEL")
        if (
            context_window_tokens <= 0
            or max_completion_tokens <= 0
            or timeout_seconds <= 0
            or progress_interval_seconds <= 0
        ):
            raise ValueError("Copilot token limits and timeout must be positive")

        self.executable = executable
        self.model = model
        self.context = context
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = max_completion_tokens
        self.timeout_seconds = timeout_seconds
        self.progress_interval_seconds = progress_interval_seconds
        self.use_proxy = use_proxy
        self.cli_log_level = cli_log_level
        self.cli_log_dir = cli_log_dir
        self._call_count = 0
        self.identity = (
            f"copilot_cli:executable={executable}:model={model}:context={context}:"
            f"context_window_tokens={context_window_tokens}:"
            f"max_completion_tokens={max_completion_tokens}:"
            f"timeout_seconds={timeout_seconds:g}"
        )

    @classmethod
    def from_environment(cls, settings: Mapping[str, str]) -> "CopilotCLIAdapter":
        return cls(
            executable=settings.get("COPILOT_CLI_EXECUTABLE", "copilot"),
            model=settings.get("COPILOT_MODEL", "gpt-6-luna"),
            context=settings.get("COPILOT_CONTEXT", "long_context"),
            context_window_tokens=_positive_int(
                settings, "COPILOT_CONTEXT_WINDOW_TOKENS", 1_000_000
            ),
            max_completion_tokens=_positive_int(
                settings, "COPILOT_OUTPUT_BUDGET_TOKENS", 15_000
            ),
            timeout_seconds=_positive_float(settings, "COPILOT_TIMEOUT_SECONDS", 600),
            progress_interval_seconds=_positive_float(
                settings, "COPILOT_PROGRESS_INTERVAL_SECONDS", 15
            ),
            use_proxy=settings.get("COPILOT_USE_PROXY", "true").strip().lower()
            not in {"0", "false", "no", "off"},
            cli_log_level=settings.get("COPILOT_CLI_LOG_LEVEL", "").strip().lower(),
            cli_log_dir=settings.get("COPILOT_CLI_LOG_DIR", "").strip(),
        )

    def _command(self) -> list[str]:
        return [
            self.executable,
            "-s",
            "--output-format=text",
            "--model",
            self.model,
            "--context",
            self.context,
            "--no-custom-instructions",
            "--no-ask-user",
            "--disable-builtin-mcps",
            "--available-tools=ask_user",
            "--deny-tool=read,write,shell,url,memory,ask_user",
        ]

    def generate(self, prompt: str) -> str:
        self._call_count += 1
        call_number = self._call_count
        stage = _copilot_stage(prompt)
        environment = os.environ.copy()
        if not self.use_proxy:
            for name in (
                "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                "http_proxy", "https_proxy", "all_proxy",
            ):
                environment.pop(name, None)
        environment.update(
            {
                "COPILOT_ALLOW_ALL": "false",
                "COPILOT_CUSTOM_INSTRUCTIONS_DIRS": "",
                "GITHUB_COPILOT_PROMPT_MODE_EXTENSIONS": "false",
                "GITHUB_COPILOT_PROMPT_MODE_REPO_HOOKS": "false",
                "GITHUB_COPILOT_PROMPT_MODE_WORKSPACE_MCP": "false",
            }
        )
        prompt_tokens = len(
            tiktoken.get_encoding(
                os.getenv("CONDENSATION_TOKENIZER_ENCODING", "o200k_base")
            ).encode(prompt)
        )
        command = self._command()
        if self.cli_log_level:
            if self.cli_log_dir:
                log_directory = Path(self.cli_log_dir).expanduser().resolve()
                log_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            else:
                log_directory = Path(tempfile.mkdtemp(prefix="podcast-copilot-"))
            command.extend([
                "--log-level", self.cli_log_level,
                "--log-dir", str(log_directory),
            ])
            print(
                f"[INFO] [COPILOT] Request #{call_number} native CLI logs: "
                f"{log_directory} (level={self.cli_log_level}; "
                "may contain sensitive data; redact before sharing)"
            )
        resolved_executable = shutil.which(self.executable)
        if resolved_executable:
            command[0] = resolved_executable
        token_names = ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")
        auth_source = next(
            (name for name in token_names if environment.get(name)),
            "CLI saved login",
        )
        network_names = (
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
            "http_proxy", "https_proxy", "all_proxy", "no_proxy",
            "NODE_EXTRA_CA_CERTS", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE",
        )
        active_network_settings = [name for name in network_names if environment.get(name)]
        debug_enabled = os.getenv("COPILOT_DEBUG", "").strip().lower() in {
            "1", "true", "yes", "on"
        }
        print(
            f"[INFO] [COPILOT] Request #{call_number} started: stage={stage} "
            f"model={self.model} context={self.context} "
            f"prompt_chars={len(prompt)} prompt_tokens={prompt_tokens} "
            f"stdin=yes timeout={self.timeout_seconds:g}s"
        )
        print(
            f"[INFO] [COPILOT] executable={command[0]!r} auth_source={auth_source} "
            f"proxy_mode={'inherited' if self.use_proxy else 'disabled for CLI child'} "
            f"network_env_names={active_network_settings} "
            f"config_home={'custom' if environment.get('COPILOT_HOME') else 'default'}"
        )
        if debug_enabled:
            print(f"[DEBUG] [COPILOT] Request #{call_number} argv={command!r}")
            print(
                f"[DEBUG] [COPILOT] Request #{call_number} exact stdin prompt "
                f"(chars={len(prompt)}) follows:\n{prompt}\n"
                f"[DEBUG] [COPILOT] Request #{call_number} prompt end"
            )
        started_at = time.monotonic()
        progress_stop = threading.Event()

        def report_progress() -> None:
            while not progress_stop.wait(self.progress_interval_seconds):
                elapsed = time.monotonic() - started_at
                print(
                    f"[INFO] [COPILOT] Request #{call_number} still running: "
                    f"stage={stage} elapsed={elapsed:.1f}s "
                    f"timeout={self.timeout_seconds:g}s "
                    "(CLI process pending; inference progress unknown)"
                )

        progress_thread = threading.Thread(target=report_progress, daemon=True)
        progress_thread.start()
        try:
            result = subprocess.run(
                command,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
                shell=False,
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - started_at
            print(
                f"[ERROR] [COPILOT] Request #{call_number} stage={stage} timed out after {elapsed:.1f}s "
                f"(limit={self.timeout_seconds:g}s)"
            )
            raise LLMAdapterError(
                f"Copilot CLI timed out after {self.timeout_seconds:g} seconds"
            ) from exc
        except OSError as exc:
            print(f"[ERROR] [COPILOT] Could not start executable {self.executable!r}: {exc}")
            raise LLMAdapterError(
                f"Could not start Copilot CLI executable {self.executable!r}"
            ) from exc
        finally:
            progress_stop.set()
            progress_thread.join()

        elapsed = time.monotonic() - started_at
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        print(
            f"[INFO] [COPILOT] Request #{call_number} finished: stage={stage} "
            f"exit={result.returncode} "
            f"elapsed={elapsed:.1f}s stdout_chars={len(stdout)} stderr_chars={len(stderr)}"
        )
        if stderr and (debug_enabled or result.returncode != 0 or not stdout.strip()):
            diagnostic = _redact_cli_diagnostic(stderr)
            print(f"[DEBUG] [COPILOT] Request #{call_number} stderr follows:\n{diagnostic}")
        if result.returncode != 0:
            hint = _cli_diagnostic_hint(stderr)
            print(f"[ERROR] [COPILOT] Non-zero exit diagnostic category: {hint}")
            raise LLMAdapterError(
                f"Copilot CLI exited with status {result.returncode} ({hint})"
            )
        output = stdout.strip()
        if not output:
            hint = _cli_diagnostic_hint(stderr)
            print(f"[ERROR] [COPILOT] Empty stdout diagnostic category: {hint}")
            raise LLMAdapterError(
                f"Copilot CLI returned empty output (exit={result.returncode}; {hint})"
            )
        return _validate_output_budget(output, self.max_completion_tokens)


def create_condensation_adapter(
    model: Any,
    model_key: str,
    environ: Mapping[str, str] | None = None,
) -> LMStudioAdapter | CopilotCLIAdapter:
    """Create the configured adapter; LM Studio remains the default."""
    settings = os.environ if environ is None else environ
    adapter_type = settings.get("CONDENSATION_ADAPTER", "lm_studio").strip().lower()

    if adapter_type == "lm_studio":
        base_url = settings.get("LM_STUDIO_BASE_URL", "http://localhost:1234/v1").rstrip("/")
        context_window_tokens = _positive_int(
            settings, "LM_STUDIO_CONTEXT_WINDOW_TOKENS", 32_768
        )
        max_completion_tokens = _positive_int(
            settings, "LM_STUDIO_MAX_COMPLETION_TOKENS", 15_000
        )
        return LMStudioAdapter(
            model=model,
            model_key=model_key,
            base_url=base_url,
            context_window_tokens=context_window_tokens,
            max_completion_tokens=max_completion_tokens,
        )

    if adapter_type == "copilot_cli":
        return CopilotCLIAdapter.from_environment(settings)

    raise ValueError(
        "CONDENSATION_ADAPTER must be 'lm_studio' or 'copilot_cli'"
    )