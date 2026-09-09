from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from workbench.asr import MemorySnapshot, qwen_memory_safe, qwen_memory_status, qwen_model_for_quality
from workbench.config import ConfigError, _parse_toml, load_config


class ConfigTests(unittest.TestCase):
    def test_repository_defaults_include_experimental_qwen_candidate(self) -> None:
        config = load_config(user_path=Path("/definitely/missing/nextecho-config.toml"))
        self.assertEqual(config["asr"]["accurate_model"], "aufklarer/Qwen3-ASR-1.7B-MLX-5bit")
        self.assertEqual(config["hardware"]["reserve_memory_gb"], 3.0)
        self.assertEqual(config.source_label, "repo-default")

    def test_user_override_is_deep_merged_and_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "config.toml"
            path.write_text('[asr]\nlanguage = "en"\n[hardware]\nreserve_memory_gb = 4.0\n', encoding="utf-8")
            config = load_config(user_path=path)
        self.assertEqual(config["asr"]["language"], "en")
        self.assertEqual(config["asr"]["fast_model"], "Qwen/Qwen3-ASR-0.6B")
        self.assertEqual(config["hardware"]["reserve_memory_gb"], 4.0)
        self.assertEqual(config.source_label, "user-override")
        self.assertTrue(any(item.endswith("config.toml") for item in config.source_paths))

    def test_fallback_parser_does_not_evaluate_values(self) -> None:
        text = '[asr]\nmodel = "Qwen/#candidate"\nmodels = ["a", "b"]\nflag = true\n'
        with patch.dict("sys.modules", {"tomllib": None, "tomli": None}):
            parsed = _parse_toml(text, source="test")
        self.assertEqual(parsed["asr"]["model"], "Qwen/#candidate")
        self.assertEqual(parsed["asr"]["models"], ["a", "b"])
        self.assertIs(parsed["asr"]["flag"], True)

    def test_invalid_toml_is_wrapped_in_config_error(self) -> None:
        with self.assertRaises(ConfigError):
            _parse_toml("[broken\nvalue = true", source="test")


class QwenMemoryGateTests(unittest.TestCase):
    def test_existing_cumulative_swap_does_not_veto_gate(self) -> None:
        snapshot = MemorySnapshot(available_bytes=4 * 1024**3, pressure="normal", swap_used_bytes=900 * 1024**2, captured_at="now")
        with patch("workbench.asr.is_apple_silicon", return_value=True):
            self.assertTrue(qwen_memory_safe(snapshot=snapshot))

    def test_swap_delta_and_reserve_are_enforced(self) -> None:
        before = MemorySnapshot(available_bytes=6 * 1024**3, pressure="normal", swap_used_bytes=0, captured_at="before")
        after = MemorySnapshot(available_bytes=4 * 1024**3, pressure="normal", swap_used_bytes=2 * 1024**3, captured_at="after")
        with patch("workbench.asr.is_apple_silicon", return_value=True):
            result = qwen_memory_status(snapshot=after, baseline=before)
            self.assertFalse(result["safe"])
            self.assertIn("swap_delta_exceeded", result["reasons"])

            low_memory = qwen_memory_status(available_memory_gb=2.0, pressure="normal", snapshot=after)
            self.assertFalse(low_memory["safe"])
            self.assertIn("reserve_memory_not_met", low_memory["reasons"])

    def test_accurate_model_comes_from_central_config(self) -> None:
        self.assertEqual(qwen_model_for_quality("accurate"), "aufklarer/Qwen3-ASR-1.7B-MLX-5bit")


if __name__ == "__main__":
    unittest.main()
