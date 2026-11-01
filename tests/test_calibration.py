import json
import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from workbench.calibration import calibrate
from workbench.config import NextEchoConfig
from workbench.contracts import PipelineError
from workbench.resources import ResourcePolicy, run_owned


@unittest.skipUnless(os.name == "posix", "POSIX controlled calibration")
class CalibrationTests(unittest.TestCase):
    def test_runaway_calibration_is_stopped_and_does_not_promote_candidate(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            model = root / "model"
            model.mkdir()
            (model / "model.int8.onnx").write_bytes(b"FAKE calibration model")
            (model / "tokens.txt").write_text("fake")
            vad = root / "vad.onnx"
            vad.write_bytes(b"FAKE VAD")
            fake = root / "fake-python"
            fake.write_text(f"#!{sys.executable}\nimport sys,time\nif '-c' in sys.argv: print('{{}}')\nelse: time.sleep(20)\n")
            fake.chmod(0o755)
            audio = root / "short.wav"
            with wave.open(str(audio), "wb") as f:
                f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                f.writeframes(b"\0\0" * 16000)
            config = NextEchoConfig({"models": {"sensevoice-int8": {"path": str(model), "worker_python": str(fake)}}, "asr": {"vad_model": str(vad)}})
            def bounded(command, **kwargs):
                kwargs["sampler"] = lambda: {"available_bytes": 10**12, "pressure": "normal", "swap_used_bytes": 0}
                return run_owned(command, **kwargs)
            with patch.dict(os.environ, {"NEXTECHO_RUNTIME_DIR": str(root / "runtime")}), patch("workbench.local_asr.model_policy", return_value=ResourcePolicy(1, 10**10, .15, reserve_bytes=0)), patch("workbench.local_asr.run_owned", side_effect=bounded):
                with self.assertRaises(PipelineError):
                    calibrate(config, "sensevoice-int8", audio, root / "calibration")
            report = json.loads((root / "calibration/calibration.json").read_text())
            self.assertEqual(report["release_status"], "candidate")
            self.assertEqual(report["error_code"], "STAGE_TIMEOUT")
            self.assertTrue((root / "calibration/logs/asr.resources.json").is_file())

    def test_hung_environment_probe_stops_before_asr_and_does_not_promote(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            model = root / "model"
            model.mkdir()
            (model / "model.int8.onnx").write_bytes(b"FAKE calibration model")
            (model / "tokens.txt").write_text("fake")
            vad = root / "vad.onnx"
            vad.write_bytes(b"FAKE VAD")
            fake = root / "fake-python"
            fake.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(20)\n")
            fake.chmod(0o755)
            audio = root / "short.wav"
            with wave.open(str(audio), "wb") as f:
                f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                f.writeframes(b"\0\0" * 16000)
            config = NextEchoConfig({"models": {"sensevoice-int8": {"path": str(model), "worker_python": str(fake)}}, "asr": {"vad_model": str(vad)}})
            with patch.dict(os.environ, {"NEXTECHO_RUNTIME_DIR": str(root / "runtime")}), patch(
                "workbench.media_commands.model_policy", return_value=ResourcePolicy(1, 10**10, .15)
            ), patch("workbench.media_commands.sample_resources", return_value={"available_bytes":10**12,"pressure":"normal","swap_used_bytes":0}), patch("workbench.local_asr.run_owned") as inference:
                with self.assertRaises(PipelineError) as error:
                    calibrate(config, "sensevoice-int8", audio, root / "calibration")
            self.assertEqual(error.exception.code, "STAGE_TIMEOUT")
            inference.assert_not_called()
            report = json.loads((root / "calibration/calibration.json").read_text())
            self.assertEqual(report["release_status"], "candidate")
            self.assertEqual(report["error_code"], "STAGE_TIMEOUT")
            self.assertTrue(list((root / "runtime/media-logs").glob("*.resources.json")))
            self.assertFalse((root / "calibration/logs/asr.resources.json").exists())
