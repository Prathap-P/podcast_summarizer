import random
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

import audio_config
import phonon_stream
import phonon_tts
import phonon_voices

CATALOG = {
    "zoey": "Zoey",
    "juniper": "Juniper",
    "harper": "Harper",
    "delia": "Delia",
}
POOL = "Zoey,Juniper,Harper,Delia"


def configure(voice="", voice_file="", pool=POOL):
    """Patch the three voice settings and the catalog for one test."""
    return (
        patch.object(audio_config, "PHONON_VOICE", voice),
        patch.object(audio_config, "PHONON_VOICE_FILE", voice_file),
        patch.object(audio_config, "PHONON_VOICE_POOL", pool),
        patch.object(phonon_voices, "_catalog_names", return_value=CATALOG),
    )


class VoiceTestCase(unittest.TestCase):
    def setUp(self):
        phonon_voices.reset_state()
        self.addCleanup(phonon_voices.reset_state)

    def configure(self, **kwargs):
        for patcher in configure(**kwargs):
            patcher.start()
            self.addCleanup(patcher.stop)


class PoolParsingTests(VoiceTestCase):
    def test_no_voice_names_are_hardcoded_in_code(self):
        with patch.dict("os.environ", {}, clear=False):
            import importlib
            import os

            os.environ.pop("PHONON_VOICE_POOL", None)
            reloaded = importlib.reload(audio_config)
            try:
                self.assertEqual(reloaded.PHONON_VOICE_POOL, "")
            finally:
                importlib.reload(audio_config)

    def test_a_voice_added_to_the_setting_is_used_without_code_changes(self):
        catalog = dict(CATALOG, newvoice="NewVoice")
        patchers = (
            patch.object(audio_config, "PHONON_VOICE", ""),
            patch.object(audio_config, "PHONON_VOICE_FILE", ""),
            patch.object(
                audio_config, "PHONON_VOICE_POOL", POOL + ",NewVoice"
            ),
            patch.object(phonon_voices, "_catalog_names", return_value=catalog),
        )
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        picks = {phonon_voices.select_voice() for _ in range(80)}
        self.assertIn("NewVoice", picks)
        self.assertEqual(len(picks), 5)
        self.assertIn("NewVoice", phonon_voices.describe())

    def test_empty_pool_tells_the_user_how_to_add_voices(self):
        self.configure(pool="")
        self.assertIn("PHONON_VOICE_POOL", phonon_voices.describe())
        self.assertIn(".env", phonon_voices.describe())

    def test_parse_accepts_commas_semicolons_newlines_and_any_case(self):
        self.configure()
        self.assertEqual(
            phonon_voices.parse_voice_pool(" zoey, JUNIPER ,harper;\ndelia "),
            ["Zoey", "Juniper", "Harper", "Delia"],
        )

    def test_voice_names_with_spaces_and_accents_are_supported(self):
        catalog = dict(
            CATALOG,
            **{
                phonon_voices._key("Master Aldric"): "Master Aldric",
                phonon_voices._key("Lucía-Sol"): "Lucía-Sol",
            },
        )
        with patch.object(phonon_voices, "_catalog_names", return_value=catalog):
            self.assertEqual(
                phonon_voices.parse_voice_pool(
                    "Zoey,  master   aldric , lucia-sol"
                ),
                ["Zoey", "Master Aldric", "Lucía-Sol"],
            )

    def test_two_unknown_words_are_one_unknown_name(self):
        self.configure()
        with patch("builtins.print") as printed:
            names = phonon_voices.parse_voice_pool("Zoey, Typo Voice")
        self.assertEqual(names, ["Zoey"])
        warnings = [c for c in printed.call_args_list if "Typo Voice" in str(c)]
        self.assertEqual(len(warnings), 1)

    def test_parse_drops_duplicates_and_unknown_names(self):
        self.configure()
        names = phonon_voices.parse_voice_pool("Zoey,zoey,Nobody,Harper")
        self.assertEqual(names, ["Zoey", "Harper"])

    def test_parse_blank_gives_empty_pool(self):
        self.configure()
        self.assertEqual(phonon_voices.parse_voice_pool("  "), [])

    def test_parse_keeps_names_when_catalog_is_unavailable(self):
        with patch.object(phonon_voices, "_catalog_names", return_value=None):
            self.assertEqual(
                phonon_voices.parse_voice_pool("Zoey,Custom"), ["Zoey", "Custom"]
            )


class UnknownVoiceWarningTests(VoiceTestCase):
    def test_unknown_voice_warns_only_once(self):
        self.configure()
        with patch("builtins.print") as printed:
            for _ in range(5):
                phonon_voices.parse_voice_pool("Zoey,Nobody")
        warnings = [c for c in printed.call_args_list if "Nobody" in str(c)]
        self.assertEqual(len(warnings), 1)


class PickTests(unittest.TestCase):
    def test_never_repeats_previous_voice(self):
        names = ["A", "B", "C", "D"]
        rng = random.Random(1)
        previous = None
        for _ in range(200):
            chosen = phonon_voices.pick_voice(names, previous, rng)
            self.assertNotEqual(chosen, previous)
            previous = chosen

    def test_every_voice_gets_picked_eventually(self):
        names = ["A", "B", "C", "D"]
        rng = random.Random(7)
        seen, previous = set(), None
        for _ in range(100):
            previous = phonon_voices.pick_voice(names, previous, rng)
            seen.add(previous)
        self.assertEqual(seen, set(names))

    def test_single_voice_pool_may_repeat(self):
        self.assertEqual(phonon_voices.pick_voice(["A"], "A"), "A")

    def test_empty_pool_gives_blank(self):
        self.assertEqual(phonon_voices.pick_voice([]), "")


class SelectVoiceTests(VoiceTestCase):
    def test_rotates_between_pool_voices_without_repeats(self):
        self.configure()
        picks = [phonon_voices.select_voice() for _ in range(40)]
        self.assertTrue(all(p in CATALOG.values() for p in picks))
        self.assertTrue(all(a != b for a, b in zip(picks, picks[1:])))
        self.assertEqual(set(picks), set(CATALOG.values()))

    def test_cloned_voice_file_disables_rotation(self):
        self.configure(voice_file="/some/voice.safetensors")
        self.assertEqual(phonon_voices.mode(), "cloned")
        self.assertEqual(phonon_voices.select_voice(), "")

    def test_pinned_voice_disables_rotation(self):
        self.configure(voice="Zoey")
        self.assertEqual(phonon_voices.mode(), "pinned")
        self.assertEqual(phonon_voices.select_voice(), "")

    def test_cloned_file_wins_over_pinned_voice(self):
        self.configure(voice="Zoey", voice_file="/some/voice.safetensors")
        self.assertEqual(phonon_voices.mode(), "cloned")

    def test_empty_pool_means_default_voice(self):
        self.configure(pool="")
        self.assertEqual(phonon_voices.mode(), "default")
        self.assertEqual(phonon_voices.select_voice(), "")

    def test_unavailable_voices_are_skipped(self):
        self.configure()
        for name in ("Juniper", "Harper", "Delia"):
            phonon_voices.mark_unavailable(name)
        self.assertEqual(
            {phonon_voices.select_voice() for _ in range(10)}, {"Zoey"}
        )

    def test_all_unavailable_falls_back_to_default(self):
        self.configure()
        for name in CATALOG.values():
            phonon_voices.mark_unavailable(name)
        self.assertEqual(phonon_voices.select_voice(), "")

    def test_describe_explains_each_mode(self):
        self.configure()
        self.assertIn("rotate", phonon_voices.describe())
        self.assertIn("Zoey, Juniper, Harper, Delia", phonon_voices.describe())
        for patcher in configure(voice="Harper"):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.assertIn("pinned", phonon_voices.describe())
        self.assertIn("Clear PHONON_VOICE", phonon_voices.describe())

    def test_fallback_order_tries_chosen_then_others_then_default(self):
        self.configure()
        order = phonon_voices.fallback_order("Harper")
        self.assertEqual(order[0], "Harper")
        self.assertEqual(set(order[1:-1]), {"Zoey", "Juniper", "Delia"})
        self.assertIsNone(order[-1])

    def test_label_for_names_the_active_voice(self):
        self.configure(voice_file="/x.safetensors")
        self.assertEqual(phonon_voices.label_for(""), "your cloned voice")
        self.assertEqual(phonon_voices.label_for("Harper"), "Harper")


def _fake_ptts(tts):
    module = types.ModuleType("ptts")
    module.DEFAULT_MODEL = "unused"
    module.TTS = Mock(return_value=tts)
    return module


class GenerateWithPickedVoiceTests(VoiceTestCase):
    def _tts(self, failing=()):
        tts = Mock()
        tts.sample_rate = 24000
        tts.device = "cpu"

        def synth(text, voice=None):
            if voice in failing:
                raise OSError(f"cannot fetch voice {voice}")
            return np.array([0.1, -0.1], dtype=np.float32)

        tts.synth.side_effect = synth
        return tts

    def test_picked_voice_is_used_for_the_file_path(self):
        self.configure()
        tts = self._tts()
        with patch.dict(sys.modules, {"ptts": _fake_ptts(tts)}):
            phonon_tts.generate_audio_phonon("Hello.", voice="Harper")
        tts.synth.assert_called_once_with("Hello.", voice="Harper")

    def test_model_loads_without_a_fixed_voice_when_rotating(self):
        self.configure()
        tts = self._tts()
        module = _fake_ptts(tts)
        with patch.dict(sys.modules, {"ptts": module}):
            phonon_tts.generate_audio_phonon("Hello.", voice="Delia")
        self.assertIsNone(module.TTS.call_args.kwargs["voice"])

    def test_cloned_file_beats_a_picked_voice(self):
        import tempfile
        from pathlib import Path

        tts = self._tts()
        with tempfile.TemporaryDirectory() as tmp:
            voice_file = Path(tmp) / "me.safetensors"
            voice_file.write_bytes(b"x")
            self.configure(voice_file=str(voice_file))
            with patch.dict(sys.modules, {"ptts": _fake_ptts(tts)}):
                phonon_tts.generate_audio_phonon("Hello.", voice="Harper")
        tts.synth.assert_called_once_with(
            "Hello.", voice=phonon_tts.CUSTOM_VOICE_NAME
        )

    def test_unloadable_voice_falls_back_to_another_pool_voice(self):
        self.configure()
        tts = self._tts(failing={"Harper"})
        with patch.dict(sys.modules, {"ptts": _fake_ptts(tts)}):
            audio, rate = phonon_tts.generate_audio_phonon("Hi.", voice="Harper")
        self.assertEqual(rate, 24000)
        voices_tried = [c.kwargs.get("voice") for c in tts.synth.call_args_list]
        self.assertEqual(voices_tried[0], "Harper")
        self.assertNotIn(None, voices_tried[:2])
        self.assertEqual(len(voices_tried), 2)

    def test_falls_back_to_default_voice_when_every_pool_voice_fails(self):
        self.configure()
        tts = self._tts(failing=set(CATALOG.values()))
        with patch.dict(sys.modules, {"ptts": _fake_ptts(tts)}):
            phonon_tts.generate_audio_phonon("Hi.", voice="Harper")
        last = tts.synth.call_args_list[-1]
        self.assertEqual(last, unittest.mock.call("Hi."))

    def test_pinned_voice_beats_a_stale_picked_voice(self):
        self.configure(voice="Zoey")
        tts = self._tts()
        module = _fake_ptts(tts)
        with patch.dict(sys.modules, {"ptts": module}):
            phonon_tts.generate_audio_phonon("Hello.", voice="Harper")
        self.assertEqual(module.TTS.call_args.kwargs["voice"], "Zoey")
        tts.synth.assert_called_once_with("Hello.")

    def test_without_a_picked_voice_behaviour_is_unchanged(self):
        self.configure(voice="Zoey")
        tts = self._tts()
        module = _fake_ptts(tts)
        with patch.dict(sys.modules, {"ptts": module}):
            phonon_tts.generate_audio_phonon("Hello.")
        self.assertEqual(module.TTS.call_args.kwargs["voice"], "Zoey")
        tts.synth.assert_called_once_with("Hello.")


def _stream_ctx(chunks):
    ctx = Mock()
    ctx.__enter__ = Mock(return_value=iter(chunks))
    ctx.__exit__ = Mock(return_value=False)
    return ctx


class LiveStreamVoiceTests(VoiceTestCase):
    def setUp(self):
        super().setUp()
        patcher = patch.object(phonon_stream, "_usable_voices", set())
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch.object(phonon_stream, "_voice", None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_stream_uses_the_resolved_voice_for_every_piece(self):
        self.configure()
        tts = Mock()
        tts.stream.side_effect = lambda piece, voice=None: _stream_ctx(
            [np.array([0.5], dtype=np.float32)]
        )
        text = ("A" * 200 + ". ") + ("B" * 200 + ".")
        with patch.object(phonon_stream, "_get_tts", return_value=tts):
            list(phonon_stream.stream_pcm16(text, "Juniper"))
        voices = [c.kwargs.get("voice") for c in tts.stream.call_args_list]
        self.assertEqual(voices, ["Juniper", "Juniper"])

    def test_resolve_returns_a_loadable_voice_and_remembers_it(self):
        self.configure()
        tts = Mock()
        tts.stream.side_effect = lambda piece, voice=None: _stream_ctx([])
        with patch.object(phonon_stream, "_get_tts", return_value=tts):
            self.assertEqual(phonon_stream.resolve_voice("Harper"), "Harper")
            self.assertEqual(phonon_stream.resolve_voice("Harper"), "Harper")
        self.assertEqual(tts.stream.call_count, 1)

    def test_resolve_skips_failing_voice_and_marks_it_unavailable(self):
        self.configure()
        tts = Mock()

        def stream(piece, voice=None):
            if voice == "Harper":
                raise OSError("403 Forbidden")
            return _stream_ctx([])

        tts.stream.side_effect = stream
        with patch.object(phonon_stream, "_get_tts", return_value=tts):
            resolved = phonon_stream.resolve_voice("Harper")
        self.assertNotEqual(resolved, "Harper")
        self.assertIn(resolved, CATALOG.values())
        self.assertNotIn("Harper", {phonon_voices.select_voice() for _ in range(20)})

    def test_resolve_gives_default_when_nothing_loads(self):
        self.configure()
        tts = Mock()
        tts.stream.side_effect = OSError("403 Forbidden")
        with patch.object(phonon_stream, "_get_tts", return_value=tts):
            self.assertEqual(phonon_stream.resolve_voice("Harper"), "")

    def test_cloned_voice_is_never_resolved_or_replaced(self):
        self.configure()
        tts = Mock()
        tts.stream.side_effect = lambda piece, voice=None: _stream_ctx([])
        with (
            patch.object(phonon_stream, "_voice", "custom"),
            patch.object(phonon_stream, "_get_tts", return_value=tts),
        ):
            self.assertEqual(phonon_stream.resolve_voice("Harper"), "")
            list(phonon_stream.stream_pcm16("Hi there.", "Harper"))
        self.assertEqual(tts.stream.call_args.kwargs["voice"], "custom")


class VoiceRouteTests(VoiceTestCase):
    @classmethod
    def setUpClass(cls):
        import app as flask_app

        cls.flask_app = flask_app
        cls.client = flask_app.app.test_client()

    def test_status_describes_the_voice_mode_for_phonon(self):
        self.configure()
        with patch.object(self.flask_app, "TTS_BACKEND", "phonon"):
            info = self.client.get("/tts_status").get_json()
        self.assertIn("Zoey, Juniper, Harper, Delia", info["voice_info"])

    def test_status_has_no_voice_info_for_kokoro(self):
        with patch.object(self.flask_app, "TTS_BACKEND", "kokoro"):
            info = self.client.get("/tts_status").get_json()
        self.assertEqual(info["voice_info"], "")

    def test_stream_reports_the_chosen_voice_in_a_header(self):
        self.configure()
        with (
            patch.object(self.flask_app, "TTS_BACKEND", "phonon"),
            patch.object(phonon_stream, "get_sample_rate", return_value=24000),
            patch.object(
                phonon_stream, "resolve_voice", side_effect=lambda v: v
            ),
            patch.object(
                phonon_stream, "stream_pcm16", return_value=iter([b"\x01\x00"])
            ) as stream,
        ):
            response = self.client.post("/stream_audio", json={"text": "Hi."})
            response.get_data()
        self.assertIn(response.headers["X-Voice"], CATALOG.values())
        self.assertEqual(stream.call_args.args[1], response.headers["X-Voice"])

    def test_text_to_audio_passes_a_picked_voice_to_the_worker(self):
        self.configure()
        with (
            patch.object(self.flask_app, "TTS_BACKEND", "phonon"),
            patch.object(
                self.flask_app, "run_in_subprocess", return_value="out/a.wav"
            ) as run,
        ):
            response = self.client.post(
                "/text_to_audio", json={"text": "Hello there."}
            )
        self.assertEqual(response.status_code, 200)
        args = run.call_args.args
        self.assertEqual(args[1:3], ("Hello there.", ""))
        self.assertIn(args[3], CATALOG.values())

    def test_other_backends_get_no_phonon_voice(self):
        with patch.object(self.flask_app, "TTS_BACKEND", "kokoro"):
            self.assertEqual(self.flask_app._pick_phonon_voice(), "")


if __name__ == "__main__":
    unittest.main()
