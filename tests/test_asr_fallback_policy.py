import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from workbench.asr import TranscriptResult
from workbench.config import NextEchoConfig
from workbench.contracts import PipelineError
from workbench.transcription import annotate_whisper_transcript, transcribe_audio_with_fallback
from workbench.io_utils import write_json


class FallbackPolicyTests(unittest.TestCase):
    def invoke(self, root, *, allow, engine="fun-asr-nano", error="MODEL_MISSING"):
        settings = NextEchoConfig({"asr": {"allow_fallback": allow, "strict_selection": True},
                     "models": {"sensevoice-int8": {"path": "FAKE-SENSE"}}})
        result = TranscriptResult("测试文字", "zh", [], "sensevoice", "FAKE-SENSE", "fake-version",
                    provenance={"sha256": "fake-sense-digest"}, segments=[{"text": "测试文字", "start_ms": 0, "end_ms": 1000}])
        calls = []
        def asr(*args, **kwargs):
            calls.append(kwargs["engine"])
            if kwargs["engine"] == "fun-asr-nano":
                raise PipelineError(error, "fake primary failure", blocked=error == "MODEL_MISSING")
            return result
        with patch("workbench.local_asr.transcribe_local_audio", side_effect=asr), patch("workbench.transcription.cached_model", return_value=None), patch("workbench.transcription.resolve_whisper_model", side_effect=AssertionError("download resolver called")):
            actual = transcribe_audio_with_fallback(audio_path=root / "audio.wav", output_dir=root, output_base="transcript",
                     whisper_bin="fake-whisper", whisper_model_path=None, whisper_model_name="tiny",
                     language="zh", output_formats=["txt", "json"], engine=engine, model="FAKE-PRIMARY", config=settings)
        return actual, calls

    def test_explicit_fallback_records_actual_identity_without_whisper(self):
        with tempfile.TemporaryDirectory() as d:
            actual, calls = self.invoke(Path(d), allow=True)
            self.assertEqual(calls, ["fun-asr-nano", "sensevoice"])
            self.assertEqual(actual["requested_engine"], "fun-asr-nano")
            self.assertEqual(actual["actual_engine"], "sensevoice")
            self.assertEqual(actual["provenance"]["sha256"], "fake-sense-digest")
            self.assertTrue(actual["degraded"])
            self.assertEqual(actual["fallbacks"][0]["outcome"], "selected")

    def test_disabled_fallback_and_cancellation_propagate(self):
        with tempfile.TemporaryDirectory() as d:
            for allow, error in ((False, "MODEL_MISSING"), (True, "CANCELLED")):
                with self.assertRaises(PipelineError) as caught:
                    self.invoke(Path(d), allow=allow, error=error)
                self.assertEqual(caught.exception.code, error)

    def test_missing_whisper_can_use_allowed_cpu_backend_without_download(self):
        with tempfile.TemporaryDirectory() as d:
            actual, calls = self.invoke(Path(d), allow=True, engine="whisper-cpp")
            self.assertEqual(calls, ["sensevoice"])
            self.assertEqual(actual["actual_engine"], "sensevoice")
            self.assertEqual(actual["requested_engine"], "whisper-cpp")

    def test_raw_transcript_content_is_never_diagnostic_redacted(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "transcript.json"
            text = "原文示例 token=spoken-value https://host/path?example=123"
            write_json(path, {"transcription": [{"text": text}]})
            annotate_whisper_transcript(path, model="test", language="zh")
            import json
            self.assertEqual(json.loads(path.read_text())["transcription"][0]["text"], text)
