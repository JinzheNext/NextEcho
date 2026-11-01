import tempfile
import unittest
from pathlib import Path

from workbench.config import NextEchoConfig, load_config
from workbench.contracts import PipelineError, PipelineRequest, plan_request
from workbench.io_utils import atomic_write, redact


class ContractTests(unittest.TestCase):
    def test_quality_plan_uses_27b_for_polish(self):
        p = plan_request(PipelineRequest(("file.wav",), profile="quality"), load_config())
        self.assertEqual(p.effective["resolved"]["polish_model"], "qwen3.8-27b-local")
        self.assertEqual(p.request.postprocess, "polish")
        self.assertEqual(p.request.speaker_mode, "off")

    def test_explicit_request_and_config_snapshot(self):
        c = NextEchoConfig({})
        p = plan_request(PipelineRequest(("a.wav",), asr_engine="whisper-cpp", asr_model="small"), c)
        self.assertEqual(p.effective["resolved"]["asr_model"], "small")
        self.assertEqual(p.effective["resolved"]["asr_engine"], "whisper-cpp")
        self.assertNotIn("resolved", c)

    def test_invalid_input_is_rejected(self):
        for r in (PipelineRequest(()), PipelineRequest(("x",), limit=21), PipelineRequest(("x",), max_seconds=-1)):
            with self.assertRaises(PipelineError):
                plan_request(r, load_config())

    def test_explicit_engine_selects_matching_profile_and_model(self):
        sense = plan_request(PipelineRequest(("x",), profile="quality", asr_engine="sensevoice"), load_config())
        self.assertEqual(sense.asr_profile, "sensevoice-int8")
        whisper = plan_request(PipelineRequest(("x",), asr_engine="whisper-cpp"), load_config())
        self.assertEqual(whisper.effective["resolved"]["asr_model"], "large-v3-turbo-q5_0")

    def test_atomic_private_file_and_redaction(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x"
            atomic_write(p, "内容")
            self.assertEqual(p.read_text(), "内容")
            self.assertEqual(p.stat().st_mode & 0o777, 0o600)
        self.assertNotIn("secret", redact("https://host/a?sig=secret token=secret"))
