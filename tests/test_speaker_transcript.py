from __future__ import annotations

import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from workbench.speaker_transcript import (
    AudioWindow,
    DiarizationTurn,
    TranscriptSegment,
    TranscriptWord,
    SpeakerTextTurn,
    align_segments_to_speakers,
    build_audio_windows,
    build_speaker_transcript,
    detect_intro_range,
    apply_context_speaker_names,
    resolve_speaker_backend,
    render_speaker_text,
    run_diarization,
)
from workbench.diarization import (
    DiarizationBackendUnavailable,
    DiarizationResult,
    SegmentClusteringBackend,
    SherpaOnnxBackend,
    render_rttm,
    write_raw_diarization_outputs,
)
from workbench.speaker_identity import DefaultSpeakerIdentityResolver


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

    def test_complete_mengyan_lijigang_rounds_do_not_leak_text_anchor(self) -> None:
        candidates = {"speaker_candidates": [{"name": "孟岩"}, {"name": "李继刚"}]}
        segments = [
            TranscriptSegment(0.0, 2.0, "大家好，我是李继刚。", "seg_00001"),
            TranscriptSegment(2.0, 4.0, "孟岩先聊聊背景。", "seg_00002"),
            TranscriptSegment(4.0, 6.0, "李继刚我补充一点。", "seg_00003"),
            TranscriptSegment(6.0, 8.0, "孟岩再回应一下。", "seg_00004"),
            TranscriptSegment(8.0, 10.0, "我是李继刚，最后总结。", "seg_00005"),
        ]
        diarization = [
            DiarizationTurn("Speaker 1", 0.0, 2.0, confidence=0.9),
            DiarizationTurn("Speaker 2", 2.0, 4.0, confidence=0.9),
            DiarizationTurn("Speaker 1", 4.0, 6.0, confidence=0.9),
            DiarizationTurn("Speaker 2", 6.0, 8.0, confidence=0.9),
            DiarizationTurn("Speaker 1", 8.0, 10.0, confidence=0.9),
        ]
        aligned, unassigned = align_segments_to_speakers(segments, diarization, 0.0, context=candidates)
        self.assertEqual(unassigned, [])
        self.assertEqual([turn.raw_speaker for turn in aligned], ["Speaker 1", "Speaker 2", "Speaker 1", "Speaker 2", "Speaker 1"])
        # The compatibility aligner only applies an anchor to its own text;
        # later turns retain the acoustic cluster until identity resolution.
        self.assertEqual(aligned[2].speaker, "Speaker 1")
        self.assertEqual(aligned[3].speaker, "Speaker 2")

        identities = DefaultSpeakerIdentityResolver().resolve(diarization, segments, context=candidates)
        by_label = identities.by_raw_label()
        self.assertEqual(by_label["Speaker 1"].display_name, "李继刚")
        self.assertEqual(by_label["Speaker 1"].identity_status, "resolved")
        self.assertEqual(by_label["Speaker 2"].display_name, "unknown")
        self.assertEqual(by_label["Speaker 2"].identity_status, "unknown")

    def test_word_alignment_splits_one_asr_segment_at_speaker_switch(self) -> None:
        segment = TranscriptSegment(
            0.0,
            4.0,
            "甲乙丙丁。",
            "seg_switch",
            words=[
                TranscriptWord("甲乙", 0.0, 1.0),
                TranscriptWord("丙丁", 2.1, 3.0),
                TranscriptWord("。", 3.0, 3.2),
            ],
        )
        aligned, unassigned = align_segments_to_speakers(
            [segment],
            [DiarizationTurn("SPEAKER_00", 0.0, 2.0), DiarizationTurn("SPEAKER_01", 2.0, 4.0)],
            0.0,
        )
        self.assertEqual(unassigned, [])
        self.assertEqual([(item.speaker, item.text) for item in aligned], [("SPEAKER_00", "甲乙"), ("SPEAKER_01", "丙丁。")])
        self.assertLess(aligned[0].end, aligned[1].start)

    def test_raw_diarization_artifacts_are_immutable_and_rttm_is_anonymous(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = DiarizationResult(
                backend="resemblyzer",
                model="resemblyzer/VoiceEncoder",
                turns=[DiarizationTurn("孟岩", 1.0, 2.5, confidence=0.4)],
            )
            json_path, rttm_path = write_raw_diarization_outputs(root, result, file_id="episode")
            first_json = json_path.read_text(encoding="utf-8")
            self.assertEqual(json.loads(first_json)["turns"][0]["raw_speaker_label"], "孟岩")
            self.assertIn("孟岩", rttm_path.read_text(encoding="utf-8"))
            result.turns[0].end = 9.0
            with self.assertRaises(FileExistsError):
                write_raw_diarization_outputs(root, result, file_id="episode")
            self.assertEqual(json_path.read_text(encoding="utf-8"), first_json)

    def test_backend_contract_can_be_loaded_without_optional_runtime(self) -> None:
        backend = SegmentClusteringBackend()
        self.assertEqual(backend.name, "segment-clustering")
        self.assertIn("SPEAKER", render_rttm(DiarizationResult("fallback", [DiarizationTurn("SPEAKER_00", 0, 1)])))

    def test_explicit_backend_and_config_are_resolved_without_auto_override(self) -> None:
        requested, selected, strict, options = resolve_speaker_backend(
            "sherpa-onnx",
            config={
                "diarization": {
                    "backend": "auto",
                    "sherpa": {"model_paths": ["models/custom/seg.onnx", "models/custom/emb.onnx"]},
                }
            },
            backend_status={"selected_backend": "resemblyzer"},
        )
        self.assertEqual((requested, selected, strict), ("sherpa-onnx", "sherpa-onnx", True))
        self.assertEqual(options["model_paths"], ["models/custom/seg.onnx", "models/custom/emb.onnx"])

    def test_sherpa_unavailable_is_explicit_failure_not_fallback(self) -> None:
        backend = SherpaOnnxBackend()
        with patch("workbench.diarization.importlib.import_module", side_effect=ModuleNotFoundError("sherpa_onnx")):
            with self.assertRaises(DiarizationBackendUnavailable) as raised:
                backend.diarize(Path("/tmp/not-used.wav"), options={"backend_config": {}})
        self.assertIn("sherpa_onnx", str(raised.exception))

    def test_sherpa_adapter_uses_configured_models_and_normalized_audio(self) -> None:
        class FakeDiarizer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def process(self, samples):
                self.sample_count = len(samples)
                return [{"speaker": "SPEAKER_00", "start": 0.0, "end": 0.5, "confidence": 0.77}]

        class FakeSherpa:
            OfflineSpeakerDiarization = FakeDiarizer

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = root / "audio.wav"
            with wave.open(str(audio), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\x00\x00" * 1600)
            segmentation = root / "segmentation.onnx"
            embedding = root / "embedding.onnx"
            segmentation.write_bytes(b"model")
            embedding.write_bytes(b"model")
            with patch("workbench.diarization.importlib.import_module", return_value=FakeSherpa()):
                result = SherpaOnnxBackend().diarize(
                    audio,
                    options={"backend_config": {"model_paths": [str(segmentation), str(embedding)]}},
                )
        self.assertEqual(result.backend, "sherpa-onnx")
        self.assertEqual(result.turns[0].raw_label, "SPEAKER_00")
        self.assertEqual(result.turns[0].confidence, 0.77)

    def test_strict_run_diarization_does_not_call_fallback_after_backend_error(self) -> None:
        class FailingBackend:
            def diarize(self, *args, **kwargs):
                raise DiarizationBackendUnavailable("missing sherpa model")

        with patch("workbench.speaker_transcript.create_diarization_backend", return_value=FailingBackend()) as factory:
            with self.assertRaises(DiarizationBackendUnavailable):
                run_diarization(
                    Path("/tmp/not-used.wav"),
                    transcript_segments=[TranscriptSegment(0, 1, "测试")],
                    intro_end=0,
                    min_speakers=2,
                    max_speakers=4,
                    backend_status={"selected_backend": "sherpa-onnx", "fallback_backend_available": True},
                    backend_config={},
                    strict_backend=True,
                )
        factory.assert_called_once_with("sherpa-onnx")


if __name__ == "__main__":
    unittest.main()
