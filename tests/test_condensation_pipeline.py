import os
import re
import unittest
from unittest.mock import patch

import tiktoken

import condenser_service


class FakeAdapter:
    identity = "fake-test-adapter"
    context_window_tokens = 12000
    max_completion_tokens = 2000

    def __init__(self, response_factory=None):
        self.prompts = []
        self.response_factory = response_factory or self._default_response

    def _default_response(self, prompt, call_index):
        if "INPUT SEGMENT:" in prompt:
            return "<final_script>mapped</final_script>"
        return "<final_script>reduced</final_script>"

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response_factory(prompt, len(self.prompts))


class CondensationPipelineTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {
            "CONDENSATION_TOKENIZER_ENCODING": "o200k_base",
            "CONDENSATION_ONE_PASS_MAX_INPUT_TOKENS": "100000",
            "CONDENSATION_MAX_INPUT_TOKENS": "300",
            "CONDENSATION_CHUNK_OVERLAP_TOKENS": "10",
            "CONDENSATION_REDUCE_BATCH_SIZE": "3",
            "CONDENSATION_FINAL_CONSOLIDATION_THRESHOLD_TOKENS": "100000",
            "CONDENSATION_PROMPT_RESERVE_TOKENS": "256",
            "CONDENSATION_CONTEXT_CONTINUITY_TOKENS": "512",
            "CONDENSATION_COMPLETION_SAFETY_TOKENS": "100",
            "CONDENSATION_MAP_OUTPUT_RATIO": "0.4",
            "CONDENSATION_REDUCE_OUTPUT_RATIO": "1.0",
            "CONDENSATION_MAX_RETRIES_PER_STEP": "3",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.encoding = tiktoken.get_encoding("o200k_base")

    def _assert_prompts_fit(self, adapter):
        for prompt in adapter.prompts:
            self.assertLessEqual(
                len(self.encoding.encode(prompt)) + adapter.max_completion_tokens,
                adapter.context_window_tokens,
            )

    def test_small_input_uses_one_pass_and_validates_final_script(self):
        adapter = FakeAdapter(lambda _prompt, _index: "<final_script>direct summary</final_script>")

        result = condenser_service.condense_content("A concise source passage.", adapter)

        self.assertEqual(result, "direct summary")
        self.assertEqual(len(adapter.prompts), 1)
        self.assertIn("HIGH-DENSITY NARRATIVE MAPPING", adapter.prompts[0])
        self._assert_prompts_fit(adapter)

    def test_one_pass_limit_respects_reserved_output_budget(self):
        os.environ["CONDENSATION_ONE_PASS_MAX_INPUT_TOKENS"] = "100000"
        os.environ["CONDENSATION_MAX_INPUT_TOKENS"] = "100000"
        adapter = FakeAdapter()

        condenser_service.condense_content("word " * 5000, adapter)

        map_calls = [prompt for prompt in adapter.prompts if "INPUT SEGMENT:" in prompt]
        self.assertGreater(len(map_calls), 1)
        self._assert_prompts_fit(adapter)

    def test_profile_id_is_stable_and_captures_adapter_settings(self):
        adapter = FakeAdapter()

        profile_id = condenser_service.get_condensation_profile_id(adapter)

        self.assertEqual(profile_id, condenser_service.get_condensation_profile_id(adapter))
        self.assertIn(adapter.identity, profile_id)
        self.assertIn('"prompt_version":"narrative-map-reduce-v1"', profile_id)
        self.assertIn('"context_window_tokens":12000', profile_id)

    def test_intro_prompt_samples_final_script_when_map_results_are_unavailable(self):
        os.environ["CONDENSATION_INTRO_MAX_INPUT_TOKENS"] = "90"
        adapter = FakeAdapter()
        final_script = "BEGIN " + "north " * 100 + "CENTER " + "south " * 100 + "ENDING"

        prompt = condenser_service.build_intro_prompt(adapter, [], final_script)

        self.assertIn("BEGIN", prompt)
        self.assertIn("CENTER", prompt)
        self.assertIn("ENDING", prompt)
        self.assertLessEqual(
            len(self.encoding.encode(prompt)) + adapter.max_completion_tokens + 256,
            adapter.context_window_tokens,
        )

    def test_intro_prompt_bounds_map_samples_to_configured_budget(self):
        os.environ["CONDENSATION_INTRO_MAX_INPUT_TOKENS"] = "90"
        adapter = FakeAdapter()
        map_samples = [f"section-{index} " * 200 for index in range(3)]

        prompt = condenser_service.build_intro_prompt(adapter, map_samples, "unused")

        self.assertIn("section-0", prompt)
        self.assertIn("section-2", prompt)
        self.assertLessEqual(
            len(self.encoding.encode(prompt)) + adapter.max_completion_tokens + 256,
            adapter.context_window_tokens,
        )

    def test_large_input_falls_back_to_map_reduce(self):
        os.environ["CONDENSATION_ONE_PASS_MAX_INPUT_TOKENS"] = "1"
        os.environ["CONDENSATION_MAX_INPUT_TOKENS"] = "45"
        adapter = FakeAdapter()
        source = "alpha beta gamma delta " * 150

        result = condenser_service.condense_content(source, adapter)

        map_calls = [prompt for prompt in adapter.prompts if "INPUT SEGMENT:" in prompt]
        reduce_calls = [prompt for prompt in adapter.prompts if "INPUT SEGMENT:" not in prompt]
        self.assertGreater(len(map_calls), 1)
        self.assertGreater(len(reduce_calls), 1)
        self.assertEqual(result.splitlines()[0], "reduced")
        self.assertLess(adapter.prompts.index(map_calls[-1]), adapter.prompts.index(reduce_calls[0]))
        self._assert_prompts_fit(adapter)

    def test_token_split_preserves_order_and_overlap(self):
        words = [f"word{index:03}" for index in range(80)]
        content = " ".join(words) + " cafe\u0301 \U0001f399 end"

        chunks = condenser_service.split_content(content, max_input_tokens=20, overlap_tokens=4)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(self.encoding.encode(chunk)), 20)
        flattened = []
        seen = set()
        for chunk in chunks:
            for word in re.findall(r"word\d+", chunk):
                if word not in seen:
                    flattened.append(word)
                    seen.add(word)
        self.assertEqual(flattened, words)
        self.assertTrue(any(chunks[index].split()[-1] in chunks[index + 1].split() for index in range(len(chunks) - 1)))
        self.assertIn("cafe\u0301", " ".join(chunks))

    def test_oversized_combined_reduce_output_is_recursively_consolidated(self):
        os.environ["CONDENSATION_ONE_PASS_MAX_INPUT_TOKENS"] = "1"
        reduce_calls = 0

        def response(prompt, _index):
            nonlocal reduce_calls
            if "INPUT SEGMENT:" in prompt:
                return f"<final_script>{'mapped ' * 180}</final_script>"
            reduce_calls += 1
            if reduce_calls <= 3:
                return f"<final_script>{'reduced ' * 160}</final_script>"
            return "<final_script>compressed result</final_script>"

        adapter = FakeAdapter(response)
        source = "source " * 700

        result = condenser_service.condense_content(source, adapter)

        self.assertIn("compressed result", result)
        self.assertGreater(reduce_calls, 3)
        self._assert_prompts_fit(adapter)

    def test_malformed_tagged_output_fails_and_records_retry(self):
        adapter = FakeAdapter(lambda _prompt, _index: "missing final tags")
        checkpoint = {
            "map_chunks": None,
            "map_results": {},
            "map_retry_counts": {},
            "reduce_results": {},
            "reduce_retry_counts": {},
            "tts_results": {},
        }

        with patch("condenser_service.save_checkpoint") as save:
            with self.assertRaisesRegex(ValueError, "Failed to remove thinking tokens"):
                condenser_service.condense_content("Short input", adapter, "fake-key", checkpoint)

        self.assertEqual(checkpoint["map_retry_counts"].get("direct"), 1)
        save.assert_called()

    def test_checkpoint_resume_reuses_completed_map_results(self):
        os.environ["CONDENSATION_ONE_PASS_MAX_INPUT_TOKENS"] = "1"
        os.environ["CONDENSATION_MAX_INPUT_TOKENS"] = "45"

        class FailFirstReduce(FakeAdapter):
            def __init__(self):
                super().__init__()
                self.failed = False

            def generate(self, prompt):
                self.prompts.append(prompt)
                if "INPUT SEGMENT:" not in prompt and not self.failed:
                    self.failed = True
                    raise RuntimeError("simulated interruption")
                return self._default_response(prompt, len(self.prompts))

        checkpoint = {
            "map_chunks": None,
            "map_results": {},
            "map_retry_counts": {},
            "reduce_batches_total": None,
            "reduce_results": {},
            "reduce_retry_counts": {},
            "tts_results": {},
        }
        source = "alpha beta gamma delta " * 150
        interrupted = FailFirstReduce()

        with patch("condenser_service.save_checkpoint"):
            with self.assertRaisesRegex(ValueError, "Model crashed during REDUCE"):
                condenser_service.condense_content(source, interrupted, "fake-key", checkpoint)
            mapped_count = len(checkpoint["map_results"])
            resumed = FakeAdapter()
            result = condenser_service.condense_content(source, resumed, "fake-key", checkpoint)

        self.assertGreater(mapped_count, 1)
        self.assertFalse(any("INPUT SEGMENT:" in prompt for prompt in resumed.prompts))
        self.assertEqual(result.split("\n\n"), ["reduced"] * len(checkpoint["reduce_results"]))
        self.assertIn("0", checkpoint["reduce_results"])


if __name__ == "__main__":
    unittest.main()