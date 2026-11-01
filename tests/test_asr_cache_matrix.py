import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from workbench.config import NextEchoConfig
from workbench.io_utils import write_json
from workbench.sources import resolve_sources
from workbench.transcription import transcribe_single_source


class AsrCacheMatrixTests(unittest.TestCase):
    def test_only_asr_dependencies_invalidate_asr_and_staged_downstream(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "input.wav"
            source.write_bytes(b"original media")
            model = root / "model"
            model.mkdir()
            (model / "model.int8.onnx").write_bytes(b"first weights")
            (model / "tokens.txt").write_text("first tokens")
            vad = root / "vad.onnx"
            vad.write_bytes(b"first VAD")
            descriptor = resolve_sources([str(source)])[0]
            config = NextEchoConfig({"runtime": {"resume_verified": True}, "asr": {"vad_model": str(vad)}})
            request = dict(descriptor=descriptor, items_dir=root / "staged/items", model_path=None, whisper_bin=None,
                ffmpeg_bin="fake-ffmpeg", language="zh", max_seconds=0, output_formats=["txt", "json"],
                glossary_text="", enable_enhancement=False, ai_polish=False, ai_style="", index=1,
                asr_engine="sensevoice", asr_model=str(model), config=config, defer_enhancement=True)
            calls = []
            normalizations = []
            version = ["1"]
            def normalize(**kwargs):
                normalizations.append(kwargs["max_seconds"])
                kwargs["target_path"].write_bytes(b"normalized audio")
            def asr(**kwargs):
                calls.append(kwargs["model"])
                (kwargs["output_dir"] / "transcript.txt").write_text("FAKE transcript")
                write_json(kwargs["output_dir"] / "transcript.json", {"transcription": []})
                return {"actual_engine": "sensevoice", "actual_model": kwargs["model"]}
            def runtime(*args):
                return {"versions": {"fake-runtime": version[0]}, "available": True}
            with patch("workbench.transcription.extract_audio", side_effect=normalize), patch("workbench.transcription.transcribe_audio_with_fallback", side_effect=asr), patch("workbench.capabilities.probe_environment", side_effect=runtime):
                first = transcribe_single_source(**request)
                self.assertEqual(first["status"], "ready")
                self.assertEqual(transcribe_single_source(**request)["status"], "cached")
                request.update(glossary_text="new glossary", ai_style="new style", postprocess_profile="quality")
                self.assertEqual(transcribe_single_source(**request)["status"], "cached")
                self.assertEqual(len(calls), 1)
                self.assertEqual(len(normalizations), 1)
                changes = [lambda: request.update(language="en"), lambda: request.update(max_seconds=1),
                           lambda: source.write_bytes(b"changed media"), lambda: (model / "model.int8.onnx").write_bytes(b"other weights"),
                           lambda: (model / "tokens.txt").write_text("other tokens"), lambda: version.__setitem__(0, "2"),
                           lambda: Path(first["audio_path"]).write_bytes(b"corrupted normalized audio"),
                           lambda: request.update(config=NextEchoConfig({"runtime": {"resume_verified": True}, "asr": {"vad_model": str(vad), "output_script": "simplified"}})),
                           lambda: request.update(config=NextEchoConfig({"runtime": {"resume_verified": True}, "asr": {"vad_model": str(vad), "output_script": "simplified", "hotwords": ["专名"]}}))]
                for count, change in enumerate(changes, 2):
                    change()
                    result = transcribe_single_source(**request)
                    self.assertEqual(result["status"], "ready", json.dumps(result))
                    self.assertEqual(len(calls), count)
                # Language, ASR weights, tokens and runtime changes reuse
                # normalized audio; range/content/corrupt audio must re-extract.
                self.assertEqual(len(normalizations), 4)

    def test_failed_asr_resume_reuses_verified_remote_media_and_audio(self):
        from workbench.contracts import PipelineError
        from workbench.sources import SourceDescriptor
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "download.wav"
            source.write_bytes(b"downloaded public media")
            descriptor = SourceDescriptor("url", "generic", "https://example.org/media.wav", "https://example.org/media.wav",
                "https://example.org/media.wav", "episode", "", 1, "", "001_episode", "direct")
            request = dict(descriptor=descriptor, items_dir=root / "items", model_path=None, whisper_bin=None,
                ffmpeg_bin="fake", language="zh", max_seconds=0, output_formats=["txt", "json"], glossary_text="",
                enable_enhancement=False, ai_polish=False, ai_style="", index=1, asr_engine="sensevoice",
                config=NextEchoConfig({"runtime": {"resume_verified": True}}))
            def normalize(**kwargs):
                kwargs["target_path"].write_bytes(b"audio")
            def complete(**kwargs):
                (kwargs["output_dir"] / "transcript.txt").write_text("restored")
                write_json(kwargs["output_dir"] / "transcript.json", {"transcription": []})
                return {"actual_engine": "sensevoice"}
            with patch("workbench.transcription.prepare_media_source", return_value=source) as download, patch("workbench.transcription.extract_audio", side_effect=normalize) as extract, patch("workbench.capabilities.probe_environment", return_value={}):
                with patch("workbench.transcription.transcribe_audio_with_fallback", side_effect=PipelineError("WORKER_FAILED", "test")):
                    self.assertEqual(transcribe_single_source(**request)["status"], "failed")
                with patch("workbench.transcription.transcribe_audio_with_fallback", side_effect=complete):
                    self.assertEqual(transcribe_single_source(**request)["status"], "ready")
                self.assertEqual(download.call_count, 1)
                self.assertEqual(extract.call_count, 1)
