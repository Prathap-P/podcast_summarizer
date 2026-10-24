import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

import audio_config
import model_worker
import phonon_tts


class PhononTtsTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(audio_config, "PHONON_VOICE_FILE", "")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_generate_audio_uses_default_checkpoint_and_voice(self):
        fake_tts = Mock()
        fake_tts.sample_rate = 24000
        fake_tts.synth.return_value = np.array([0.1, -0.1], dtype=np.float64)
        fake_ptts = types.ModuleType("ptts")
        fake_ptts.DEFAULT_MODEL = "hf://gradium/phonon-31m-en@revision"
        fake_ptts.TTS = Mock(return_value=fake_tts)

        with (
            patch.dict(sys.modules, {"ptts": fake_ptts}),
            patch.object(audio_config, "PHONON_MODEL_ID", ""),
            patch.object(audio_config, "PHONON_LANG", "en"),
            patch.object(audio_config, "PHONON_VOICE", ""),
        ):
            audio, sample_rate = phonon_tts.generate_audio_phonon(
                "Hello from Phonon."
            )

        fake_ptts.TTS.assert_called_once_with(
            model=None, device=None, lang="en", voice=None
        )
        fake_tts.synth.assert_called_once_with("Hello from Phonon.")
        self.assertEqual(audio.dtype, np.float32)
        np.testing.assert_array_equal(
            audio, np.array([0.1, -0.1], dtype=np.float32)
        )
        self.assertEqual(sample_rate, 24000)

    def test_generate_audio_uses_configured_model_language_and_voice(self):
        fake_tts = Mock()
        fake_tts.sample_rate = 24000
        fake_tts.synth.return_value = np.array([0.2], dtype=np.float32)
        fake_ptts = types.ModuleType("ptts")
        fake_ptts.DEFAULT_MODEL = "unused"
        fake_ptts.TTS = Mock(return_value=fake_tts)

        with (
            patch.dict(sys.modules, {"ptts": fake_ptts}),
            patch.object(
                audio_config, "PHONON_MODEL_ID", "hf://example/model"
            ),
            patch.object(audio_config, "PHONON_LANG", "fr"),
            patch.object(audio_config, "PHONON_VOICE", "Zoey"),
        ):
            audio, sample_rate = phonon_tts.generate_audio_phonon("Bonjour.")

        fake_ptts.TTS.assert_called_once_with(
            model="hf://example/model",
            device=None,
            lang="fr",
            voice="Zoey",
        )
        self.assertEqual(audio.shape, (1,))
        self.assertEqual(sample_rate, 24000)

    def test_generate_audio_rejects_invalid_pcm(self):
        fake_tts = Mock()
        fake_tts.sample_rate = 24000
        fake_tts.synth.return_value = np.zeros((2, 4), dtype=np.float32)
        fake_ptts = types.ModuleType("ptts")
        fake_ptts.DEFAULT_MODEL = "unused"
        fake_ptts.TTS = Mock(return_value=fake_tts)

        with (
            patch.dict(sys.modules, {"ptts": fake_ptts}),
            patch.object(audio_config, "PHONON_MODEL_ID", ""),
            patch.object(audio_config, "PHONON_LANG", "en"),
            patch.object(audio_config, "PHONON_VOICE", ""),
            self.assertRaisesRegex(ValueError, "mono PCM"),
        ):
            phonon_tts.generate_audio_phonon("Hello.")

    def test_generate_audio_registers_and_uses_custom_voice_file(self):
        fake_tts = Mock()
        fake_tts.sample_rate = 24000
        fake_tts.synth.return_value = np.array([0.3], dtype=np.float32)
        fake_ptts = types.ModuleType("ptts")
        fake_ptts.DEFAULT_MODEL = "unused"
        fake_ptts.TTS = Mock(return_value=fake_tts)

        with tempfile.TemporaryDirectory() as tmp:
            voice_file = Path(tmp) / "myvoice.safetensors"
            voice_file.write_bytes(b"x")
            with (
                patch.dict(sys.modules, {"ptts": fake_ptts}),
                patch.object(audio_config, "PHONON_MODEL_ID", ""),
                patch.object(audio_config, "PHONON_LANG", "en"),
                patch.object(audio_config, "PHONON_VOICE", "Zoey"),
                patch.object(audio_config, "PHONON_VOICE_FILE", str(voice_file)),
            ):
                phonon_tts.generate_audio_phonon("Hello.")

        fake_ptts.TTS.assert_called_once_with(
            model=None, device=None, lang="en", voice=None
        )
        fake_tts.add_voice.assert_called_once_with(
            phonon_tts.CUSTOM_VOICE_NAME, voice_file
        )
        fake_tts.synth.assert_called_once_with(
            "Hello.", voice=phonon_tts.CUSTOM_VOICE_NAME
        )

    def test_generate_audio_rejects_missing_voice_file(self):
        fake_ptts = types.ModuleType("ptts")
        fake_ptts.DEFAULT_MODEL = "unused"
        fake_ptts.TTS = Mock()

        with (
            patch.dict(sys.modules, {"ptts": fake_ptts}),
            patch.object(audio_config, "PHONON_MODEL_ID", ""),
            patch.object(audio_config, "PHONON_LANG", "en"),
            patch.object(audio_config, "PHONON_VOICE", ""),
            patch.object(audio_config, "PHONON_VOICE_FILE", "/nonexistent/v.safetensors"),
            self.assertRaises(FileNotFoundError),
        ):
            phonon_tts.generate_audio_phonon("Hello.")

        fake_ptts.TTS.assert_not_called()


class TtsWorkerTests(unittest.TestCase):
    def test_worker_dispatches_phonon_and_preserves_sample_rate(self):
        audio = np.array([0.1, -0.1], dtype=np.float32)
        create_audio_file = Mock(return_value="kokoro_outputs/video-title.wav")
        fake_kokoro = types.ModuleType("kokoro_tts")
        fake_kokoro.create_audio_file = create_audio_file
        generate_audio = Mock(return_value=(audio, 24000))
        fake_phonon = types.ModuleType("phonon_tts")
        fake_phonon.generate_audio_phonon = generate_audio

        with (
            patch.dict(
                sys.modules,
                {"kokoro_tts": fake_kokoro, "phonon_tts": fake_phonon},
            ),
            patch.dict(os.environ, {"TTS_BACKEND": "phonon"}),
            patch("model_worker.load_dotenv"),
        ):
            result = model_worker.tts_worker(
                "Hello from Phonon.",
                "",
                video_id="video-id",
                video_title="video-title",
            )

        self.assertEqual(result, "kokoro_outputs/video-title.wav")
        generate_audio.assert_called_once_with("Hello from Phonon.")
        create_audio_file.assert_called_once_with(
            audio,
            sample_rate=24000,
            video_id="video-id",
            video_title="video-title",
        )

    def test_worker_writes_phonon_audio_with_the_shared_wav_writer(self):
        audio = np.array([0.1, -0.1, 0.05], dtype=np.float32)
        fake_phonon = types.ModuleType("phonon_tts")
        fake_phonon.generate_audio_phonon = Mock(return_value=(audio, 24000))
        original_cwd = Path.cwd()

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                os.chdir(temp_dir)
                with (
                    patch.dict(sys.modules, {"phonon_tts": fake_phonon}),
                    patch.dict(os.environ, {"TTS_BACKEND": "phonon"}),
                    patch("model_worker.load_dotenv"),
                ):
                    audio_path = model_worker.tts_worker(
                        "Hello from Phonon.",
                        "",
                        video_id="phonon-smoke",
                    )

                info = sf.info(audio_path)
                self.assertEqual(info.samplerate, 24000)
                self.assertEqual(info.channels, 1)
                self.assertGreater(info.frames, 0)
            finally:
                os.chdir(original_cwd)

    def test_worker_keeps_kokoro_dispatch_when_selected(self):
        audio = np.array([0.1], dtype=np.float32)
        generate_audio = Mock(return_value=audio)
        create_audio_file = Mock(return_value="kokoro_outputs/kokoro.wav")
        fake_kokoro = types.ModuleType("kokoro_tts")
        fake_kokoro.generate_audio = generate_audio
        fake_kokoro.create_audio_file = create_audio_file

        with (
            patch.dict(sys.modules, {"kokoro_tts": fake_kokoro}),
            patch.dict(
                os.environ, {"TTS_BACKEND": "kokoro", "HF_HUB_OFFLINE": "0"}
            ),
            patch("model_worker.load_dotenv"),
        ):
            result = model_worker.tts_worker("Hello from Kokoro.", "")
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")

        self.assertEqual(result, "kokoro_outputs/kokoro.wav")
        generate_audio.assert_called_once_with("Hello from Kokoro.")
        create_audio_file.assert_called_once_with(
            audio, video_id="", video_title=""
        )

    def test_worker_surfaces_phonon_errors_without_falling_back(self):
        create_audio_file = Mock()
        kokoro_generate_audio = Mock()
        fake_kokoro = types.ModuleType("kokoro_tts")
        fake_kokoro.create_audio_file = create_audio_file
        fake_kokoro.generate_audio = kokoro_generate_audio
        fake_phonon = types.ModuleType("phonon_tts")
        fake_phonon.generate_audio_phonon = Mock(
            side_effect=RuntimeError("model download failed")
        )

        with (
            patch.dict(
                sys.modules,
                {"kokoro_tts": fake_kokoro, "phonon_tts": fake_phonon},
            ),
            patch.dict(os.environ, {"TTS_BACKEND": "phonon"}),
            patch("model_worker.load_dotenv"),
            self.assertRaisesRegex(RuntimeError, "model download failed"),
        ):
            model_worker.tts_worker("Hello from Phonon.", "")

        create_audio_file.assert_not_called()
        kokoro_generate_audio.assert_not_called()
