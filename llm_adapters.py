"""Text-only adapters used by the condensation pipeline."""

import hashlib
import math
import os
import subprocess
from collections.abc import Mapping
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
    ) -> None:
        if not executable.strip():
            raise ValueError("COPILOT_CLI_EXECUTABLE must not be empty")
        if not model.strip():
            raise ValueError("COPILOT_MODEL must not be empty")
        if context not in {"default", "long_context"}:
            raise ValueError("COPILOT_CONTEXT must be 'default' or 'long_context'")
        if context_window_tokens <= 0 or max_completion_tokens <= 0 or timeout_seconds <= 0:
            raise ValueError("Copilot token limits and timeout must be positive")

        self.executable = executable
        self.model = model
        self.context = context
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = max_completion_tokens
        self.timeout_seconds = timeout_seconds
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
        environment = os.environ.copy()
        environment.update(
            {
                "COPILOT_ALLOW_ALL": "false",
                "COPILOT_CUSTOM_INSTRUCTIONS_DIRS": "",
                "GITHUB_COPILOT_PROMPT_MODE_EXTENSIONS": "false",
                "GITHUB_COPILOT_PROMPT_MODE_REPO_HOOKS": "false",
                "GITHUB_COPILOT_PROMPT_MODE_WORKSPACE_MCP": "false",
            }
        )
        try:
            result = subprocess.run(
                self._command(),
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
                shell=False,
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            raise LLMAdapterError(
                f"Copilot CLI timed out after {self.timeout_seconds:g} seconds"
            ) from exc
        except OSError as exc:
            raise LLMAdapterError(
                f"Could not start Copilot CLI executable {self.executable!r}"
            ) from exc

        if result.returncode != 0:
            raise LLMAdapterError(
                f"Copilot CLI exited with status {result.returncode}"
            )
        output = result.stdout.strip()
        if not output:
            raise LLMAdapterError("Copilot CLI returned empty output")
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