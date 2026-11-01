"""原稿/原生时间、固定本地权重和受控调用契约；不加载模型。"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from workbench.aligner_candidate import (
    MODEL_ID, REQUIRED_FILES, run_qwen_alignment_candidate,
    validate_native_alignment, verify_local_model,
)
from workbench.contracts import PipelineError
from workbench.config import NextEchoConfig
from workbench.io_utils import file_hash, write_json
from workbench.resources import GIB


class AlignerCandidateTests(unittest.TestCase):
    def model(self, root):
        root.mkdir()
        files = {}
        for name in REQUIRED_FILES:
            p = root / name
            p.write_text("synthetic fixture " + name)
            files[name] = {"bytes": p.stat().st_size, "sha256": file_hash(p)}
        return {"model_id": MODEL_ID, "revision": "a" * 40, "license": "apache-2.0", "files": files}

    def audio(self, path, frames=16000):
        with wave.open(str(path), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes(b"\0\0" * frames)
        return path

    def test_raw_text_gaps_duplicate_words_and_native_intervals(self):
        text = "你好，AI。你好，未匹配！"
        rows = [{"text": t, "start_time": s, "end_time": e}
                for t, s, e in [("你好", .1, .3), ("AI", .3, .6), ("你好", .7, .9)]]
        d = validate_native_alignment(text, rows, 1, 10000)
        self.assertEqual(d["source_text"], text)
        self.assertEqual([r["char_start"] for r in d["anchors"]], [0, 3, 6])
        self.assertEqual(d["anchors"][0]["native_start_ms"], 10100)
        self.assertEqual(d["anchors"][0]["native_end_ms"], 10300)
        self.assertEqual(d["lexical_char_coverage"], 6 / 9)
        self.assertFalse(d["precise_verified"])
        self.assertEqual(d["unanchored_source_spans"][-1]["text"], "，未匹配！")

    def test_unmatched_tokens_never_enter_source_or_gain_interpolated_times(self):
        text = "我不能删除，后缀。"
        d = validate_native_alignment(text, [{"text": "伪造", "start_time": .1, "end_time": .2}], 1)
        self.assertEqual(d["anchors"], [])
        self.assertEqual(d["unanchored_source_spans"], [{"char_start": 0, "char_end": len(text), "text": text}])
        self.assertEqual(d["unmatched_native_tokens"][0]["reason"], "no_exact_source_span")

    def test_invalid_native_times_fail_instead_of_clamping(self):
        for start, end in [(-.1, .2), (.3, .2), (0, 1.01), (float("nan"), .3), (True, .2)]:
            with self.subTest(start=start, end=end), self.assertRaises(PipelineError):
                validate_native_alignment("我", [{"text": "我", "start_time": start, "end_time": end}], 1)
        with self.assertRaises(PipelineError):
            validate_native_alignment("我是", [{"text": "我", "start_time": .5, "end_time": .6},
                                               {"text": "是", "start_time": .2, "end_time": .3}], 1)

    def test_same_onset_and_zero_duration_remain_native(self):
        d = validate_native_alignment("我是", [{"text": t, "start_time": .2, "end_time": .2} for t in "我是"], 1)
        self.assertEqual([r["native_end_ms"] for r in d["anchors"]], [200, 200])
        self.assertEqual(d["lexical_char_coverage"], 1)
        self.assertEqual(d["positive_duration_lexical_char_coverage"], 0)
        self.assertEqual(d["zero_duration_lexical_chars"], 2)
        self.assertEqual(d["max_same_time_zero_duration_native_lexical_chars"], 2)
        self.assertFalse(d["temporal_identity_evidence_verified"])

    def test_fully_matched_long_zero_width_tail_never_becomes_timing_accuracy(self):
        text = "正常" + "尾" * 93
        rows = [{"text": "正常", "start_time": .1, "end_time": .5}]
        rows += [{"text": "尾", "start_time": .7, "end_time": .7} for _ in range(93)]
        result = validate_native_alignment(text, rows, 2)
        self.assertEqual(result["lexical_char_coverage"], 1)
        self.assertEqual(result["positive_duration_lexical_char_coverage"], 2 / 95)
        self.assertEqual(result["max_same_time_zero_duration_native_lexical_chars"], 93)
        self.assertEqual(result["zero_duration_lexical_chars"], 93)
        self.assertEqual(result["anchors"][-1]["native_start_ms"], 700)
        self.assertEqual(result["anchors"][-1]["native_end_ms"], 700)
        self.assertFalse(result["precise_verified"])

    def test_zero_width_diagnostics_distinguish_punctuation_and_separate_runs(self):
        result = validate_native_alignment("甲，乙。丙丁戊", [
            {"text": "甲", "start_time": .1, "end_time": .1},
            {"text": "，", "start_time": .1, "end_time": .1},
            {"text": "乙", "start_time": .1, "end_time": .1},
            {"text": "丙", "start_time": .2, "end_time": .3},
            {"text": "丁", "start_time": .4, "end_time": .4},
            {"text": "戊", "start_time": .5, "end_time": .5}], 1)
        self.assertEqual(result["zero_duration_lexical_chars"], 4)
        self.assertEqual(result["max_same_time_zero_duration_native_lexical_chars"], 2)
        self.assertEqual(result["positive_duration_lexical_char_coverage"], 1 / 5)

    def test_positive_intervals_or_empty_lexical_reference_do_not_fake_diagnostics(self):
        result = validate_native_alignment("甲", [{"text": "甲", "start_time": .1, "end_time": .2}], 1)
        self.assertEqual(result["positive_duration_lexical_char_coverage"], 1)
        self.assertEqual(result["timing_warnings"], [])
        sub_ms = validate_native_alignment("甲", [{"text": "甲", "start_time": .10001, "end_time": .10009}], 1)
        self.assertEqual(sub_ms["anchors"][0]["native_start_ms"], sub_ms["anchors"][0]["native_end_ms"])
        self.assertEqual(sub_ms["positive_duration_lexical_char_coverage"], 1)
        self.assertEqual(sub_ms["zero_duration_lexical_chars"], 0)
        empty = validate_native_alignment("，", [], 1)
        self.assertIsNone(empty["positive_duration_lexical_char_coverage"])

    def test_local_weights_require_fixed_revision_complete_bytes_and_license(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "model"
            ledger = self.model(root)
            self.assertEqual(verify_local_model(root, ledger)["revision"], "a" * 40)
            for change in ({"revision": "main"}, {"license": "unknown"}, {"files": {}},
                           {"files": {name: [] for name in REQUIRED_FILES}}):
                with self.assertRaises(PipelineError):
                    verify_local_model(root, {**ledger, **change})
            (root / "config.json").write_text("changed")
            with self.assertRaises(PipelineError):
                verify_local_model(root, ledger)

    def test_controller_bounds_audio_and_does_not_launch_or_overwrite_existing_directory(self):
        with tempfile.TemporaryDirectory() as d, patch("workbench.aligner_candidate.run_owned") as run:
            root = Path(d)
            ledger = self.model(root / "model")
            audio = self.audio(root / "clip.wav", 300 * 16000 + 1)
            with self.assertRaises(PipelineError):
                run_qwen_alignment_candidate(python=Path(sys.executable), model_dir=root / "model",
                    ledger=ledger, audio=audio, text="我是", destination=root / "out")
            run.assert_not_called()
            self.audio(audio)
            (root / "out").mkdir()
            with self.assertRaises(FileExistsError):
                run_qwen_alignment_candidate(python=Path(sys.executable), model_dir=root / "model",
                    ledger=ledger, audio=audio, text="我是", destination=root / "out")
            run.assert_not_called()

    def test_controller_uses_shared_budget_and_checks_frozen_response(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ledger = self.model(root / "model")
            audio = self.audio(root / "clip.wav")
            def respond(command, **kwargs):
                request = json.loads(Path(command[-1]).read_text())
                self.assertEqual(request["attention"], "sdpa")
                attempt = Path(command[-1]).parent
                self.assertEqual(file_hash(attempt / "worker.source.py"), request["worker_sha256"])
                self.assertEqual(file_hash(attempt / "controller.source.py"), request["controller_sha256"])
                self.assertEqual(kwargs["policy"].reserve_bytes, 3 * GIB)
                self.assertEqual(kwargs["policy"].estimated_bytes, 8 * GIB)
                self.assertEqual(kwargs["policy"].timeout_seconds, 9)
                write_json(Path(request["output"]), {"schema_version": 1, "audio_sha256": request["audio_sha256"],
                    "text_sha256": request["text_sha256"], "model": request["model"],
                    "timestamps": [{"text": "我是", "start_time": .1, "end_time": .5}]})
                return {"samples": 2}
            with patch("workbench.aligner_candidate.run_owned", side_effect=respond):
                d = run_qwen_alignment_candidate(python=Path(sys.executable), model_dir=root / "model",
                    ledger=ledger, audio=audio, text="我是。", destination=root / "out", timeout=9)
            self.assertEqual(d["anchors"][0]["native_start_ms"], 100)
            self.assertFalse(d["production_default_changed"])
            self.assertEqual(d["source_text"], "我是。")
            def wrong_input(command, **kwargs):
                request = json.loads(Path(command[-1]).read_text())
                write_json(Path(request["output"]), {"schema_version": 1, "audio_sha256": "0" * 64})
                return {}
            with patch("workbench.aligner_candidate.run_owned", side_effect=wrong_input), self.assertRaises(PipelineError) as exc:
                run_qwen_alignment_candidate(python=Path(sys.executable), model_dir=root / "model",
                    ledger=ledger, audio=audio, text="我是。", destination=root / "wrong-input")
            self.assertEqual(exc.exception.code, "PROTOCOL_ERROR")
            def tamper(command, **kwargs):
                audio.write_bytes(b"changed")
                return {}
            with patch("workbench.aligner_candidate.run_owned", side_effect=tamper), self.assertRaises(PipelineError) as exc:
                run_qwen_alignment_candidate(python=Path(sys.executable), model_dir=root / "model",
                    ledger=ledger, audio=audio, text="我是。", destination=root / "tampered")
            self.assertEqual(exc.exception.code, "CACHE_MISMATCH")

    def test_higher_configured_resource_reserve_and_tighter_swap_are_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            ledger = self.model(root / "model")
            audio = self.audio(root / "clip.wav")
            def blocked(command, **kwargs):
                self.assertEqual(kwargs["policy"].reserve_bytes, 5 * GIB)
                self.assertEqual(kwargs["policy"].max_swap_delta_bytes, GIB // 2)
                self.assertEqual(kwargs["policy"].max_pressure, "normal")
                raise PipelineError("RESOURCE_BLOCKED", "synthetic preflight")
            with patch("workbench.aligner_candidate.run_owned", side_effect=blocked), self.assertRaises(PipelineError):
                run_qwen_alignment_candidate(python=Path(sys.executable), model_dir=root / "model", ledger=ledger,
                    audio=audio, text="我是。", destination=root / "out",
                    config=NextEchoConfig({"hardware": {"reserve_memory_gb": 5,
                        "max_swap_delta_gb": .5, "memory_pressure_max": "normal"}}))

    def test_cancel_and_invalid_settings_prevent_any_model_read_or_launch(self):
        common = dict(python=Path(sys.executable), model_dir=Path("missing-model"),
                      ledger={}, audio=Path("missing-audio"), text="我是。", destination=Path("unused"))
        with patch("workbench.aligner_candidate.run_owned") as run:
            for setting in ({"timeout": True}, {"timeout": "9"}, {"attention": "flash_attention_2"}):
                with self.assertRaises(PipelineError) as exc:
                    run_qwen_alignment_candidate(**common, **setting)
                self.assertEqual(exc.exception.code, "INVALID_INPUT")
            with self.assertRaises(PipelineError) as exc:
                run_qwen_alignment_candidate(**common, cancel=lambda: True)
            self.assertEqual(exc.exception.code, "CANCELLED")
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
