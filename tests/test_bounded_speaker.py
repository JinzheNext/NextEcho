import json
import os
import shutil
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from workbench.bounded_speaker import bounded_build_speaker_transcript
from workbench.config import NextEchoConfig
from workbench.io_utils import write_json
from workbench.contracts import PipelineError
from workbench.speaker_transcript import package_available


@unittest.skipUnless(os.name == "posix" and shutil.which("ffmpeg"), "POSIX + ffmpeg")
class BoundedSpeakerTests(unittest.TestCase):
    def test_owned_entry_requires_real_acoustic_dependencies(self):
        with tempfile.TemporaryDirectory(prefix="speaker worker 中文 ") as d:
            root = Path(d)
            with wave.open(str(root / "audio.wav"), "wb") as f:
                f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                f.writeframes(b"\0\0" * 32000)
            write_json(root / "transcript.json", {"transcription": [
                {"text": "大家好。", "offsets": {"from": 0, "to": 1000}},
                {"text": "讨论本地转写。", "offsets": {"from": 1000, "to": 2000}}]})
            config = NextEchoConfig({"hardware": {"reserve_memory_gb": 0},
                        "diarization": {"worker_python": sys.executable, "backend": "segment-clustering"}})
            with patch.dict(os.environ, {"NEXTECHO_RUNTIME_DIR": str(root / "runtime")}):
                if not (package_available("numpy") and package_available("sklearn")):
                    # A clean environment must not turn a transcript timeline
                    # into a successful one-person acoustic result.
                    with self.assertRaises(PipelineError) as error:
                        bounded_build_speaker_transcript(root, config=config, backend="segment-clustering")
                    self.assertEqual(error.exception.code, "DEPENDENCY_MISSING")
                    self.assertFalse((root / "transcript.speakers.json").exists())
                    return
                result = bounded_build_speaker_transcript(root, config=config, backend="segment-clustering")
            self.assertTrue((root / "transcript.speakers.json").is_file())
            self.assertIn("resources", result)
            self.assertGreater(result["resources"]["samples"], 0)
            self.assertIn("speaker_turns", result)
            self.assertEqual(result["speaker_count_detected"], 0)
            self.assertEqual(result["status"], "partial")
            self.assertEqual(json.loads((root / "transcript.json").read_text())["transcription"][0]["text"], "大家好。")
