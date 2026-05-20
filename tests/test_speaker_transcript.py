from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from workbench.speaker_transcript import (
    DiarizationTurn,
    TranscriptSegment,
    align_segments_to_speakers,
    build_speaker_transcript,
    detect_intro_range,
)


class SpeakerTranscriptTests(unittest.TestCase):
    def test_detect_intro_range_uses_repeated_promo(self) -> None:
        segments = [
            TranscriptSegment(0.0, 30.0, "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目"),
            TranscriptSegment(30.0, 60.0, "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目"),
            TranscriptSegment(60.0, 90.0, "正式对谈开始"),
        ]
        removed, intro_range = detect_intro_range(segments)
        self.assertTrue(removed)
        self.assertEqual(intro_range, [0.0, 60.0])

    def test_align_segments_assigns_and_merges_speakers(self) -> None:
        transcript_segments = [
            TranscriptSegment(60.0, 70.0, "你好。"),
            TranscriptSegment(70.2, 80.0, "我们开始吧。"),
            TranscriptSegment(80.1, 90.0, "好的。"),
        ]
        diarization_turns = [
            DiarizationTurn("Speaker 1", 59.0, 80.0),
            DiarizationTurn("Speaker 2", 80.0, 100.0),
        ]
        speaker_turns, unassigned = align_segments_to_speakers(transcript_segments, diarization_turns, 60.0)
        self.assertEqual(len(unassigned), 0)
        self.assertEqual(len(speaker_turns), 2)
        self.assertEqual(speaker_turns[0].speaker, "Speaker 1")
        self.assertEqual(speaker_turns[0].text, "你好。我们开始吧。")
        self.assertEqual(speaker_turns[1].speaker, "Speaker 2")

    def test_build_speaker_transcript_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio_path = root / "audio.normalized.wav"
            audio_path.write_bytes(b"fake audio")
            transcript_path = root / "transcript.zh.fixed.json"
            transcript_path.write_text(
                json.dumps(
                    {
                        "transcription": [
                            {
                                "timestamps": {"from": "00:00:00,000", "to": "00:00:30,000"},
                                "text": "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目",
                            },
                            {
                                "timestamps": {"from": "00:00:30,000", "to": "00:01:00,000"},
                                "text": "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目",
                            },
                            {
                                "timestamps": {"from": "00:01:00,000", "to": "00:01:20,000"},
                                "text": "跟吉刚录播客非常有意思。",
                            },
                            {
                                "timestamps": {"from": "00:01:20,000", "to": "00:01:40,000"},
                                "text": "大家好，我是李继刚。",
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with patch(
                "workbench.speaker_transcript.ensure_speaker_dependencies",
                return_value={
                    "pyannote_installed": True,
                    "hf_token_env": "HF_TOKEN",
                    "fallback_backend_available": True,
                    "selected_backend": "pyannote",
                    "ready": True,
                },
            ):
                with patch("workbench.speaker_transcript.enhance_voice_audio", return_value=root / "derived" / "audio.voice_enhanced.wav"):
                    with patch(
                        "workbench.speaker_transcript.run_diarization",
                        return_value=(
                            "pyannote",
                            [
                                DiarizationTurn("Speaker 1", 60.0, 80.0),
                                DiarizationTurn("Speaker 2", 80.0, 100.0),
                            ],
                        ),
                    ):
                        payload = build_speaker_transcript(root)

            self.assertTrue((root / "transcript.speakers.json").exists())
            self.assertTrue((root / "transcript.speakers.txt").exists())
            self.assertTrue((root / "transcript.speakers.md").exists())
            self.assertTrue((root / "speaker_map.json").exists())
            self.assertTrue(payload["intro_removed"])
            self.assertEqual(payload["backend"], "pyannote")
            self.assertEqual(payload["speaker_turns"][0]["speaker"], "Speaker 1")
            self.assertIn("跟吉刚录播客非常有意思", payload["preview_text"])


if __name__ == "__main__":
    unittest.main()
