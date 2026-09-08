from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from workbench.asr import TranscriptResult
from workbench.transcription import _load_shifted_transcription_segments, write_qwen_transcript_result


class AsrTests(unittest.TestCase):
    def test_chunk_overlap_assigns_boundary_to_previous_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "chunk.json"
            path.write_text(
                json.dumps(
                    {
                        "transcription": [
                            {"offsets": {"from": 0, "to": 1200}, "text": "overlap"},
                            {"offsets": {"from": 1500, "to": 2500}, "text": "owned"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            rows = _load_shifted_transcription_segments(path, 598500, trim_before_ms=600000)
            self.assertEqual([item["text"] for item in rows], ["owned"])

    def test_qwen_result_writes_compatible_timestamped_json_and_subtitles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = TranscriptResult(
                text="你好。NextEcho 在 2026 年工作。",
                language="Chinese",
                words=[
                    {"text": "你好。", "start": 0.0, "end": 1.0},
                    {"text": "NextEcho", "start": 1.0, "end": 2.0},
                    {"text": "在", "start": 2.0, "end": 2.2},
                    {"text": "2026", "start": 2.2, "end": 2.8},
                    {"text": "年工作。", "start": 2.8, "end": 4.0},
                ],
                backend="qwen3-asr",
                model="Qwen/Qwen3-ASR-1.7B",
                version="0.3.5",
            )
            write_qwen_transcript_result(result, output_dir=root, output_base="transcript", output_formats=["txt", "json", "srt", "vtt"])
            payload = json.loads((root / "transcript.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(payload["engine"]["model"], "Qwen/Qwen3-ASR-1.7B")
            self.assertEqual(payload["transcription"][0]["timestamps"]["from"], "00:00:00,000")
            self.assertTrue((root / "transcript.srt").exists())
            self.assertTrue((root / "transcript.vtt").exists())


if __name__ == "__main__":
    unittest.main()
