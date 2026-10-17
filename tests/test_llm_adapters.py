import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from llm_adapters import (
    CopilotCLIAdapter,
    LLMAdapterError,
    create_condensation_adapter,
)


class LLMAdapterTests(unittest.TestCase):
    def test_default_adapter_wraps_langchain_model_and_exposes_contract(self):
        model = Mock()
        model.invoke.return_value = SimpleNamespace(content="Generated text")

        adapter = create_condensation_adapter(model, "local-model", environ={})

        self.assertEqual(adapter.generate("prompt"), "Generated text")
        model.invoke.assert_called_once_with("prompt")
        self.assertEqual(adapter.context_window_tokens, 32_768)
        self.assertEqual(adapter.max_completion_tokens, 15_000)
        self.assertIn("lm_studio:model=local-model", adapter.identity)
        self.assertNotIn("localhost", adapter.identity)

    def test_copilot_defaults_and_tool_restrictions(self):
        adapter = create_condensation_adapter(
            model=None,
            model_key="unused",
            environ={"CONDENSATION_ADAPTER": "copilot_cli"},
        )
        self.assertIsInstance(adapter, CopilotCLIAdapter)
        self.assertEqual(adapter.model, "gpt-6-luna")
        self.assertEqual(adapter.context, "long_context")
        self.assertEqual(adapter.context_window_tokens, 1_000_000)

        with patch("llm_adapters.subprocess.run") as run:
            run.return_value = SimpleNamespace(returncode=0, stdout="  answer\n", stderr="")
            result = adapter.generate("private prompt")

        self.assertEqual(result, "answer")
        run.assert_called_once()
        args, kwargs = run.call_args
        self.assertEqual(kwargs["input"], "private prompt")
        self.assertNotIn("private prompt", args[0])
        self.assertFalse(kwargs["shell"])
        self.assertTrue(kwargs["capture_output"])
        self.assertTrue(kwargs["text"])
        self.assertEqual(kwargs["timeout"], 600)
        self.assertIn("-s", args[0])
        self.assertIn("--model", args[0])
        self.assertIn("gpt-6-luna", args[0])
        self.assertIn("--context", args[0])
        self.assertIn("long_context", args[0])
        self.assertIn("--no-custom-instructions", args[0])
        self.assertIn("--no-ask-user", args[0])
        self.assertIn("--disable-builtin-mcps", args[0])
        self.assertIn("--available-tools=ask_user", args[0])
        self.assertIn("--deny-tool=read,write,shell,url,memory,ask_user", args[0])
        self.assertEqual(kwargs["env"]["COPILOT_ALLOW_ALL"], "false")
        self.assertEqual(kwargs["env"]["GITHUB_COPILOT_PROMPT_MODE_WORKSPACE_MCP"], "false")

    def test_copilot_settings_are_environment_configurable(self):
        adapter = CopilotCLIAdapter.from_environment(
            {
                "COPILOT_CLI_EXECUTABLE": "/opt/copilot",
                "COPILOT_MODEL": "chosen-model",
                "COPILOT_CONTEXT": "default",
                "COPILOT_CONTEXT_WINDOW_TOKENS": "64000",
                "COPILOT_OUTPUT_BUDGET_TOKENS": "4000",
                "COPILOT_TIMEOUT_SECONDS": "42",
            }
        )

        self.assertEqual(adapter.executable, "/opt/copilot")
        self.assertEqual(adapter.model, "chosen-model")
        self.assertEqual(adapter.context, "default")
        self.assertEqual(adapter.context_window_tokens, 64_000)
        self.assertEqual(adapter.max_completion_tokens, 4_000)
        self.assertEqual(adapter.timeout_seconds, 42)

    def test_copilot_nonzero_exit_does_not_expose_process_output(self):
        adapter = create_condensation_adapter(
            None, "unused", {"CONDENSATION_ADAPTER": "copilot_cli"}
        )
        with patch("llm_adapters.subprocess.run") as run:
            run.return_value = SimpleNamespace(
                returncode=7,
                stdout="prompt secret",
                stderr="prompt secret diagnostic",
            )
            with self.assertRaisesRegex(LLMAdapterError, "status 7") as error:
                adapter.generate("prompt secret")
        self.assertNotIn("prompt secret", str(error.exception))

    def test_copilot_timeout_surfaces_useful_error_without_prompt(self):
        adapter = CopilotCLIAdapter.from_environment(
            {"COPILOT_TIMEOUT_SECONDS": "3"}
        )
        with patch(
            "llm_adapters.subprocess.run",
            side_effect=subprocess.TimeoutExpired("copilot", 3),
        ):
            with self.assertRaisesRegex(LLMAdapterError, "timed out after 3 seconds"):
                adapter.generate("sensitive prompt")

    def test_copilot_empty_output_is_an_error(self):
        adapter = CopilotCLIAdapter.from_environment({})
        with patch("llm_adapters.subprocess.run") as run:
            run.return_value = SimpleNamespace(returncode=0, stdout=" \n", stderr="")
            with self.assertRaisesRegex(LLMAdapterError, "empty output"):
                adapter.generate("prompt")

    def test_copilot_output_over_budget_is_rejected(self):
        adapter = CopilotCLIAdapter.from_environment(
            {"COPILOT_OUTPUT_BUDGET_TOKENS": "1"}
        )
        with patch("llm_adapters.subprocess.run") as run:
            run.return_value = SimpleNamespace(
                returncode=0,
                stdout="<final_script>an over-budget response</final_script>",
                stderr="",
            )
            with self.assertRaisesRegex(LLMAdapterError, "exceeded the configured output budget"):
                adapter.generate("prompt")

    def test_copilot_configuration_is_validated(self):
        with self.assertRaisesRegex(ValueError, "COPILOT_CONTEXT"):
            CopilotCLIAdapter.from_environment({"COPILOT_CONTEXT": "unknown"})
        with self.assertRaisesRegex(ValueError, "COPILOT_TIMEOUT_SECONDS"):
            CopilotCLIAdapter.from_environment({"COPILOT_TIMEOUT_SECONDS": "0"})
        with self.assertRaisesRegex(ValueError, "COPILOT_TIMEOUT_SECONDS"):
            CopilotCLIAdapter.from_environment({"COPILOT_TIMEOUT_SECONDS": "nan"})


if __name__ == "__main__":
    unittest.main()