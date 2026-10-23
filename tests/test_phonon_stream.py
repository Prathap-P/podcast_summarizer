import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

import phonon_stream


class SplitTextTests(unittest.TestCase):
    def test_short_sentences_are_packed_into_chunks(self):
        chunks = phonon_stream.split_text("One. Two! Three?", max_chars=100)
        self.assertEqual(chunks, ["One. Two! Three?"])

    def test_chunks_respect_limit_and_keep_all_words(self):
        text = " ".join(f"Sentence number {i} is here." for i in range(200))
        chunks = phonon_stream.split_text(text, max_chars=120)
        self.assertGreater(len(chunks), 10)
        self.assertTrue(all(len(c) <= 120 for c in chunks))
        self.assertEqual(" ".join(chunks).split(), text.split())

    def test_long_sentence_without_punctuation_is_split(self):
        text = "word " * 500
        chunks = phonon_stream.split_text(text, max_chars=100)
        self.assertTrue(all(len(c) <= 100 for c in chunks))
        self.assertEqual(" ".join(chunks).split(), text.split())

    def test_oversized_token_is_hard_split(self):
        chunks = phonon_stream.split_text("x" * 250, max_chars=100)
        self.assertEqual([len(c) for c in chunks], [100, 100, 50])

    def test_blank_text_gives_no_chunks(self):
        self.assertEqual(phonon_stream.split_text("  \n  "), [])


def _fake_stream_ctx(chunks):
    ctx = Mock()
    ctx.__enter__ = Mock(return_value=iter(chunks))
    ctx.__exit__ = Mock(return_value=False)
    return ctx


class StreamTests(unittest.TestCase):
    def test_to_pcm16_clips_and_scales(self):
        data = phonon_stream.to_pcm16(np.array([0.0, 1.0, -1.0, 2.0]))
        samples = np.frombuffer(data, dtype="<i2")
        self.assertEqual(samples.tolist(), [0, 32767, -32767, 32767])

    def test_stream_pcm16_streams_pieces_in_order(self):
        fake_tts = Mock()
        fake_tts.stream.side_effect = lambda piece: _fake_stream_ctx(
            [np.array([0.5], dtype=np.float32)]
        )
        text = ("A" * 200 + ". ") + ("B" * 200 + ".")
        with patch.object(phonon_stream, "_get_tts", return_value=fake_tts):
            out = list(phonon_stream.stream_pcm16(text))

        self.assertEqual(
            [c.args[0][0] for c in fake_tts.stream.call_args_list], ["A", "B"]
        )
        self.assertEqual([len(b) for b in out], [2, 2])

    def test_closing_iterator_early_releases_lock(self):
        ctx = _fake_stream_ctx([np.zeros(2, dtype=np.float32)] * 5)
        fake_tts = Mock()
        fake_tts.stream.return_value = ctx
        with patch.object(phonon_stream, "_get_tts", return_value=fake_tts):
            gen = phonon_stream.stream_pcm16("Hello there.")
            next(gen)
            gen.close()
        self.assertTrue(phonon_stream._stream_lock.acquire(blocking=False))
        phonon_stream._stream_lock.release()
        ctx.__exit__.assert_called()

    def test_model_is_loaded_once(self):
        fake_tts = Mock(device="cpu", sample_rate=24000)
        fake_ptts = types.ModuleType("ptts")
        fake_ptts.TTS = Mock(return_value=fake_tts)
        with (
            patch.dict(sys.modules, {"ptts": fake_ptts}),
            patch.object(phonon_stream, "_tts", None),
        ):
            self.assertEqual(phonon_stream.get_sample_rate(), 24000)
            phonon_stream.get_sample_rate()
        fake_ptts.TTS.assert_called_once()


class LiveSpeechRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app as flask_app

        cls.flask_app = flask_app
        cls.client = flask_app.app.test_client()

    def test_status_reports_live_speech_off_for_kokoro(self):
        with patch.object(self.flask_app, "TTS_BACKEND", "kokoro"):
            info = self.client.get("/tts_status").get_json()
        self.assertFalse(info["live_speech"])
        self.assertEqual(info["backend"], "kokoro")
        self.assertIn("TTS_BACKEND=phonon", info["message"])

    def test_status_reports_live_speech_on_for_phonon(self):
        with patch.object(self.flask_app, "TTS_BACKEND", "phonon"):
            info = self.client.get("/tts_status").get_json()
        self.assertTrue(info["live_speech"])
        self.assertEqual(info["message"], "")

    def test_stream_is_refused_for_kokoro(self):
        with (
            patch.object(self.flask_app, "TTS_BACKEND", "kokoro"),
            patch.object(phonon_stream, "get_sample_rate") as load,
        ):
            response = self.client.post("/stream_audio", json={"text": "Hi."})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Phonon", response.get_json()["error"])
        load.assert_not_called()

    def test_stream_rejects_empty_text_for_phonon(self):
        with patch.object(self.flask_app, "TTS_BACKEND", "phonon"):
            response = self.client.post("/stream_audio", json={"text": " "})
        self.assertEqual(response.status_code, 400)

    def test_stream_returns_pcm_with_sample_rate_for_phonon(self):
        with (
            patch.object(self.flask_app, "TTS_BACKEND", "phonon"),
            patch.object(phonon_stream, "get_sample_rate", return_value=24000),
            patch.object(
                phonon_stream, "stream_pcm16", return_value=iter([b"\x01\x00"])
            ),
        ):
            response = self.client.post("/stream_audio", json={"text": "Hi."})
            body = response.get_data()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Sample-Rate"], "24000")
        self.assertEqual(body, b"\x01\x00")


if __name__ == "__main__":
    unittest.main()
