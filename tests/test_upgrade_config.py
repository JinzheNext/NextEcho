from __future__ import annotations

import tempfile
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from workbench.cli import _apply_config_defaults, build_parser
from workbench.config import load_config
from workbench.headless_cli import cmd_prepare
from workbench.model_profiles import PROFILES, complete_hf_snapshot, selection_decision, validate_model_files
from workbench.postprocess import resolve_postprocess_models


class UpgradeConfigTests(unittest.TestCase):
    def test_explicit_options_win_for_equals_and_separate_forms(self):
        config = load_config()
        for options in (["--asr-engine=whisper-cpp", "--postprocess-profile=off", "--language=en"],
                        ["--asr-engine", "whisper-cpp", "--postprocess-profile", "off", "--language", "en"]):
            argv = ["transcribe", "test.wav", *options]
            args = build_parser().parse_args(argv)
            _apply_config_defaults(args, config, argv)
            self.assertEqual((args.asr_engine, args.postprocess_profile, args.language), ("whisper-cpp", "off", "en"))

    def test_new_4b_is_selected_in_balanced(self):
        plan = resolve_postprocess_models("balanced", config={"postprocess": {
            "provider": "ollama", "preferred_model_id": "qwen3.5:4b-mlx"}},
            ollama_info={"models": ["qwen3.5:4b-mlx", "qwen3:4b"]})
        self.assertEqual(plan["polish_models"], ["qwen3.5:4b-mlx"])

    def test_explicit_polish_identity_wins_over_profile_and_never_falls_back(self):
        from workbench.contracts import PipelineError
        for profile, model in (("fast", "qwen3.5:9b-mlx"), ("quality", "qwen3.5:2b-mlx")):
            config = {"postprocess": {"strict_selection": True, "provider": "ollama", "preferred_model_id": model}}
            plan = resolve_postprocess_models(profile, config=config, ollama_info={"models": [model, "qwen3:4b"]})
            self.assertEqual(plan["polish_models"], [model])
            with self.assertRaises(PipelineError):
                resolve_postprocess_models(profile, config=config, ollama_info={"models": ["qwen3:4b"]})

    def test_partial_model_is_not_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "model.int8.onnx").write_bytes(b"partial")
            status = validate_model_files(root, PROFILES["sensevoice-int8"].required_files)
            self.assertFalse(status["complete"])
            self.assertEqual(status["missing_files"], ["tokens.txt"])

    def test_shipped_defaults_match_repository(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual((root / "config/nextecho.toml").read_bytes(), (root / "workbench/data/nextecho.toml").read_bytes())

    def test_indexed_snapshot_requires_all_shards(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "config.json").write_text("{}")
            (root / "tokenizer.json").write_text("{}")
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "part1.safetensors", "b": "part2.safetensors"}}))
            (root / "part1.safetensors").write_bytes(b"first")
            self.assertIsNone(complete_hf_snapshot(str(root)))
            (root / "part2.safetensors").write_bytes(b"second")
            self.assertEqual(complete_hf_snapshot(str(root)), root.resolve())

    def test_installation_binding_precedes_explicit_user_config(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "binding.json").write_text(json.dumps({"schema_version": 1, "config": {"models": {"sensevoice-int8": {"path": "bound"}}}}))
            override = root / "user.toml"
            override.write_text('[models.sensevoice-int8]\npath="explicit"\n')
            with patch.dict(os.environ, {"NEXTECHO_RUNTIME_DIR": str(root)}):
                c = load_config(override)
            self.assertEqual(c.get_path("models", "sensevoice-int8", "path"), "explicit")
            self.assertIn("installation", c.source_kinds)

    def test_loaded_snapshot_and_recorded_revision_match(self):
        from workbench.asr import transcribe_qwen_mlx
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            snapshot = root / "revision-exact"
            snapshot.mkdir()
            (snapshot / "config.json").write_text("{}")
            (snapshot / "tokenizer.json").write_text("{}")
            weights = snapshot / "model.safetensors"
            weights.write_bytes(b"weights")
            def worker(command, **kwargs):
                self.assertEqual(command[command.index("--model") + 1], str(snapshot))
                Path(command[command.index("--output") + 1]).write_text('{"text":"ok","words":[]}')
                return {}
            with patch("workbench.asr.qwen_mlx_available", return_value=True), patch("workbench.asr.qwen_memory_safe", return_value=True), patch("workbench.model_profiles.complete_hf_snapshot", return_value=snapshot), patch("workbench.resources.run_owned", side_effect=worker):
                result = transcribe_qwen_mlx(audio_path=root / "audio.wav", item_dir=root, model="owner/model", language="zh")
            self.assertEqual(result.provenance["revision"], snapshot.name)
            self.assertEqual(result.provenance["sha256"], __import__("hashlib").sha256(b"weights").hexdigest())

    def test_prepare_preserves_virtual_environment_interpreter(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            python = root / "venv/bin/python"
            python.parent.mkdir(parents=True)
            python.symlink_to(__import__("sys").executable)
            runtime = root / "runtime"
            args = build_parser().parse_args(["prepare", "--model-root", str(root / "models"), "--worker-python", str(python),
                                              "--speaker-model-root", str(root / "speaker-models")])
            with patch.dict(os.environ, {"NEXTECHO_RUNTIME_DIR": str(runtime)}), patch("builtins.print"):
                cmd_prepare(args)
            bound = json.loads((runtime / "binding.json").read_text())
            self.assertEqual(bound["config"]["models"]["sensevoice-int8"]["worker_python"], str(python))
            speaker = bound["config"]["diarization"]
            self.assertEqual(speaker["worker_python"], str(python))
            self.assertEqual(speaker["ctc_model_path"], str((root / "models/sherpa-sense-voice").resolve()))
            self.assertEqual(speaker["num_clusters"], -1)
            self.assertEqual(speaker["sherpa"]["model_paths"], [str((root / "speaker-models" / n).resolve()) for n in
                                                              ("segmentation.onnx", "embedding.onnx")])

    def test_speaker_diagnostic_requires_ctc_and_does_not_promote_quality(self):
        from workbench.capabilities import inspect_capabilities
        from workbench.config import NextEchoConfig
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            weights = [root / name for name in ("segmentation.onnx", "embedding.onnx")]
            for path in weights:
                path.write_bytes(b"fixture")
            config = NextEchoConfig({"diarization": {"sherpa": {"model_paths": [str(p) for p in weights]},
                                                    "ctc_model_path": str(root / "ctc")}})
            with patch("workbench.capabilities.probe_environment", return_value={"available": True}):
                missing = inspect_capabilities(config, capability="speaker")
                self.assertEqual(missing["capabilities"]["speaker"]["status"], "missing")
                (root / "ctc").mkdir()
                for name in PROFILES["sensevoice-int8"].required_files:
                    (root / "ctc" / name).write_bytes(b"fixture")
                available = inspect_capabilities(config, capability="speaker")
                self.assertTrue(available["ok"])
                self.assertFalse(available["production_ready"])

    def test_nano_capability_does_not_require_whisper_or_import_model(self):
        from workbench.capabilities import inspect_capabilities
        from workbench.config import NextEchoConfig
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in PROFILES["fun-asr-nano-pt"].required_files:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("fixture")
            vad = root / "vad.onnx"
            vad.write_text("fixture")
            config = NextEchoConfig({"models": {"fun-asr-nano-pt": {"path": str(root)}}, "asr": {"vad_model": str(vad)}})
            with patch("workbench.capabilities.probe_environment", return_value={"available": True, "versions": {}}), patch("workbench.capabilities.find_binary", return_value="ffmpeg") as binary:
                report = inspect_capabilities(config, profile="quality", capability="asr")
            self.assertEqual(report["capabilities"]["asr"]["status"], "ready")
            self.assertFalse(report["production_ready"])
            binary.assert_called_once_with("ffmpeg")

    def test_unverified_native_model_does_not_replace_auto(self):
        native = [{"engine": "fun-asr-nano", "model": "local", "eligible": False}]
        choice = selection_decision("auto", qwen_model="qwen", whisper_model="whisper", qwen_eligible=False, native=native)
        self.assertEqual(choice["engine"], "whisper-cpp")

    def test_changed_calibration_identity_cannot_become_auto_eligible(self):
        from workbench.model_profiles import native_asr_status
        from workbench.config import NextEchoConfig
        from workbench.io_utils import file_hash
        from workbench.capabilities import NATIVE_REQUIREMENTS
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in PROFILES["sensevoice-int8"].required_files:
                (root / name).write_bytes(b"FAKE")
            vad = root / "vad.onnx"
            vad.write_bytes(b"FAKE VAD")
            versions = {name: value or "test-version" for name, value in NATIVE_REQUIREMENTS["sensevoice-int8"].items()}
            data = {"models": {"sensevoice-int8": {"path": str(root), "estimated_bytes": 1, "max_rss_bytes": 2,
                "timeout_seconds": 3, "release_status": "ready", "runtime_versions": versions,
                "calibration_files": {name: file_hash(root / name) for name in PROFILES["sensevoice-int8"].required_files},
                "vad_sha256": file_hash(vad)}}, "asr": {"vad_model": str(vad)}}
            with patch("workbench.capabilities.probe_environment", return_value={"available": True, "versions": versions}):
                self.assertTrue(native_asr_status(NextEchoConfig(data))[1]["eligible"])
                (root / "tokens.txt").write_bytes(b"CHANGED")
                self.assertFalse(native_asr_status(NextEchoConfig(data))[1]["eligible"])


if __name__ == "__main__":
    unittest.main()
