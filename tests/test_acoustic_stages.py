import json
import os
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from workbench.acoustic_dialogue import _validate_stage, build_acoustic_dialogue, load_dialogue_segments
from workbench.config import NextEchoConfig
from workbench.contracts import PipelineError
from workbench.io_utils import file_hash, write_json


class AcousticStageTests(unittest.TestCase):
    def test_cached_audio_stages_rebuild_derived_text_and_only_changed_ctc_reruns(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            item = root / "item"
            item.mkdir()
            audio = item / "audio.wav"
            audio.write_bytes(b"stand-in audio; never decoded by this fake")
            models = [root / n for n in ("segmentation.onnx", "embedding.onnx")]
            ctc = root / "ctc"
            ctc.mkdir()
            for path in [*models, ctc / "model.int8.onnx", ctc / "tokens.txt"]:
                path.write_bytes(b"fixture")
            raw = item / "transcript.json"
            write_json(raw, {"transcription": [{"id": "s1", "text": "甲", "offsets": {"from": 0, "to": 1000}}]})
            config = NextEchoConfig({"diarization": {"worker_python": sys.executable,
                "sherpa": {"model_paths": [str(p) for p in models]}, "ctc_model_path": str(ctc)}})
            calls = []

            def owned(command, **kwargs):
                request = json.loads(Path(command[-1]).read_text())
                operation = request["operation"]
                calls.append(operation)
                clip = {"id": "full"}
                if operation == "ctc":
                    clip["tokens"] = [{"text": "甲乙", "start_ms": 100, "end_ms": 900}]
                else:
                    clip["diarization"] = {"turns": [{"start": 0, "end": 1, "speaker": "A"}]}
                write_json(Path(request["output"]), {"schema_version": 1, "clips": [clip]})
                return {"peak_rss_bytes": 123}

            probe = SimpleNamespace(returncode=0, stdout='{"python":"fake","packages":{}}')
            with patch.dict(os.environ, {"NEXTECHO_RUNTIME_DIR": str(root / "runtime")}), \
                    patch("workbench.acoustic_dialogue.run_media_command", return_value=probe), \
                    patch("workbench.resources.run_owned", side_effect=owned), \
                    patch("workbench.resources.model_policy", return_value=object()):
                build_acoustic_dialogue(item, config=config)
                self.assertEqual(calls, ["diarize-global", "ctc"])
                calls.clear()
                # A changed text layer must be realigned, without recomputing audio.
                raw.write_text(raw.read_text().replace('"甲"', '"甲乙"'))
                rebuilt = build_acoustic_dialogue(item, config=config)
                self.assertEqual(calls, [])
                self.assertEqual("".join(r["text"] for r in rebuilt["transcription"]), "甲乙")
                loaded, _ = load_dialogue_segments(item, file_hash(raw))
                self.assertEqual(loaded, rebuilt["transcription"])
                (ctc / "tokens.txt").write_bytes(b"new tokens identity")
                build_acoustic_dialogue(item, config=config)
                self.assertEqual(calls, ["ctc"])
                calls.clear()
                # A damaged cached stage must not be silently reused.
                (item / "speaker-stages/diarize-global.json").write_text("{}")
                build_acoustic_dialogue(item, config=config)
                self.assertEqual(calls, ["diarize-global"])
                calls.clear()
                (item / "speaker-stages/ctc.cache.json").write_text("broken JSON")
                build_acoustic_dialogue(item, config=config)
                self.assertEqual(calls, ["ctc"])

    def test_broken_or_slow_worker_probe_is_an_actionable_missing_dependency(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"NEXTECHO_RUNTIME_DIR": d}):
            config = NextEchoConfig({"diarization": {"worker_python": sys.executable}})
            for failure in (OSError("not executable"), subprocess.TimeoutExpired("probe", 10),
                            SimpleNamespace(returncode=0, stdout="not JSON")):
                kwargs = {"side_effect": failure} if isinstance(failure, Exception) else {"return_value": failure}
                with patch("workbench.acoustic_dialogue.run_media_command", **kwargs):
                    with self.assertRaises(PipelineError) as caught:
                        build_acoustic_dialogue(Path(d), config=config)
                self.assertEqual(caught.exception.code, "DEPENDENCY_MISSING")
                self.assertTrue(caught.exception.blocked)

    def test_wrong_clip_and_nonfinite_or_reversed_native_times_are_rejected(self):
        for clip in [{"id": "other", "tokens": []},
                     {"id": "full", "tokens": [{"text": "甲", "start_ms": float("nan")}]},
                     {"id": "full", "tokens": [{"text": "甲", "start_ms": 30, "end_ms": 20}]}]:
            with self.assertRaises(PipelineError) as caught:
                _validate_stage({"schema_version": 1, "clips": [clip]}, "ctc")
            self.assertEqual(caught.exception.code, "PROTOCOL_ERROR")
