import shutil
import subprocess
import unittest
from pathlib import Path

NODE = shutil.which("node")
SCRIPT = Path(__file__).with_name("speech_stretcher.test.js")


@unittest.skipUnless(NODE, "node is not installed")
class SpeechStretcherTests(unittest.TestCase):
    def test_speed_changes_without_changing_pitch(self):
        result = subprocess.run(
            [NODE, str(SCRIPT)], capture_output=True, text=True, timeout=120
        )
        self.assertEqual(
            result.returncode, 0, msg=result.stdout + result.stderr
        )


if __name__ == "__main__":
    unittest.main()
