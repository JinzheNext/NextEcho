from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from workbench.speaker_transcript import (
    AudioWindow,
    DiarizationTurn,
    TranscriptSegment,
    SpeakerTextTurn,
    align_segments_to_speakers,
    build_audio_windows,
    build_speaker_transcript,
    detect_intro_range,
    apply_context_speaker_names,
    render_speaker_text,
)


class SpeakerTranscriptTests(unittest.TestCase):
    def test_render_speaker_text_uses_dialogue_format_and_removes_fillers(self) -> None:
        rendered = render_speaker_text(
            [{"speaker": "孟岩", "start": 0.0, "end": 2.0, "text": "嗯大家好。"}],
            with_timestamps=False,
        )
        self.assertIn("孟岩：大家好。", rendered)

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
            self.assertEqual(payload["fallback_used"], False)
            self.assertEqual(payload["speaker_count_detected"], 2)
            self.assertEqual(payload["speaker_turns"][0]["speaker"], "Speaker 1")
            self.assertIn("跟吉刚录播客非常有意思", payload["preview_text"])
            speaker_text = (root / "transcript.speakers.txt").read_text(encoding="utf-8")
            self.assertTrue(speaker_text.startswith("本文转录由 GitHub 项目：NextEcho 提供支持"))
            self.assertIn("powered by github.com/JinzheNext/NextEcho - 让播客自动转录进知识库", speaker_text)
            speaker_map = json.loads((root / "speaker_map.json").read_text(encoding="utf-8"))
            self.assertEqual(sorted(speaker_map.keys()), ["Speaker 1", "Speaker 2"])

    def test_build_audio_windows_splits_long_segments(self) -> None:
        windows = build_audio_windows(
            [
                TranscriptSegment(10.0, 20.0, "这是一段比较长的话。"),
                TranscriptSegment(22.0, 23.0, "短句。"),
            ],
            intro_end=0.0,
            window_seconds=3.0,
            step_seconds=1.5,
        )
        self.assertGreater(len(windows), 3)
        self.assertIsInstance(windows[0], AudioWindow)
        self.assertLessEqual(windows[0].end - windows[0].start, 3.0)

    def test_context_names_are_hints_and_preserve_unmatched_labels(self) -> None:
        turns = [
            # The self-introduction is intentionally enough to identify one
            # side; the other candidate is assigned only as a reviewable hint.
            SpeakerTextTurn("Speaker 1", 0.0, 2.0, "大家好，我是李继刚。"),
            SpeakerTextTurn("Speaker 2", 2.0, 4.0, "我们开始聊聊。"),
        ]
        renamed, mapping, inference = apply_context_speaker_names(
            turns,
            context={"speaker_candidates": [{"name": "孟岩"}, {"name": "李继刚"}]},
        )
        self.assertEqual(mapping["Speaker 1"], "李继刚")
        self.assertEqual(mapping["Speaker 2"], "孟岩")
        self.assertEqual([turn.speaker for turn in renamed], ["李继刚", "孟岩"])
        self.assertEqual(inference["status"], "hinted")

    def test_text_self_introduction_corrects_early_voiceprint_switch(self) -> None:
        segments = [
            TranscriptSegment(0.0, 2.0, "跟嘉宾录播客非常有意思。"),
            TranscriptSegment(2.0, 4.0, "我们先聊聊背景。"),
            TranscriptSegment(4.0, 6.0, "大家好，我是李继刚。"),
        ]
        turns, unassigned = align_segments_to_speakers(
            segments,
            [DiarizationTurn("Speaker 2", 0.0, 6.0)],
            0.0,
            context={"speaker_candidates": [{"name": "孟岩"}, {"name": "李继刚"}]},
        )
        self.assertEqual(unassigned, [])
        self.assertEqual([turn.speaker for turn in turns], ["孟岩", "李继刚"])


if __name__ == "__main__":
    unittest.main()
