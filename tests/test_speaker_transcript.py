from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
import wave
from types import ModuleType, SimpleNamespace
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
    cluster_speaker_embeddings,
    detect_intro_range,
    exclusive_diarization_turns,
    has_lexical_content,
    apply_context_speaker_names,
    resolve_speaker_backend,
    resolve_speaker_workspace,
    render_speaker_text,
    run_diarization,
    run_pyannote_diarization,
    PYANNOTE_COMMUNITY_REVISION,
    PYANNOTE_MODEL_ID,
)
from workbench.diarization import (
    DiarizationBackendUnavailable,
    DiarizationResult,
    DiarizationOptions,
    PyannoteCommunityBackend,
    SegmentClusteringBackend,
    SherpaOnnxBackend,
    render_rttm,
    write_raw_diarization_outputs,
)
from workbench.speaker_identity import DefaultSpeakerIdentityResolver


class SpeakerTranscriptTests(unittest.TestCase):
    @staticmethod
    def _directory_digest(path: Path) -> str:
        digest = hashlib.sha256()
        for item in sorted(p for p in path.rglob("*") if p.is_file()):
            name = item.relative_to(path).as_posix().encode()
            digest.update(len(name).to_bytes(8, "big"))
            digest.update(name)
            digest.update(item.read_bytes())
        return digest.hexdigest()

    def test_community_runtime_loads_fixed_local_snapshot_and_preserves_both_tracks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot = root / PYANNOTE_COMMUNITY_REVISION
            snapshot.mkdir()
            (snapshot / "config.yaml").write_text("fixed local config", encoding="utf-8")
            calls = {}

            class Annotation:
                def __init__(self, rows):
                    self.rows = rows

                def itertracks(self, *, yield_label):
                    for start, end, label in self.rows:
                        yield SimpleNamespace(start=start, end=end), None, label

            diarization = SimpleNamespace(
                speaker_diarization=Annotation([(0.0, 1.0, "SPEAKER_00"),
                                                (0.5, 1.5, "SPEAKER_01")]),
                exclusive_speaker_diarization=Annotation([(0.0, 1.5, "SPEAKER_00")]),
            )

            class FakePipeline:
                @classmethod
                def from_pretrained(cls, model_path):
                    calls["model_path"] = model_path
                    calls["offline_environment"] = {
                        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                        "HF_HUB_DISABLE_TELEMETRY": os.environ.get("HF_HUB_DISABLE_TELEMETRY"),
                        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
                        "PYANNOTE_METRICS_ENABLED": os.environ.get("PYANNOTE_METRICS_ENABLED"),
                    }
                    return lambda audio_path, **kwargs: calls.update(
                        audio_path=audio_path, constraints=kwargs
                    ) or diarization

            package = ModuleType("pyannote")
            package.__path__ = []
            audio_module = ModuleType("pyannote.audio")
            audio_module.Pipeline = FakePipeline
            config = {
                "model_id": PYANNOTE_MODEL_ID,
                "revision": PYANNOTE_COMMUNITY_REVISION,
                "snapshot_path": str(snapshot),
                "model_digest": self._directory_digest(snapshot),
            }
            env_names = ("HF_HUB_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "TRANSFORMERS_OFFLINE", "PYANNOTE_METRICS_ENABLED")
            original_environment = {name: os.environ.get(name) for name in env_names}
            with patch.dict(sys.modules, {"pyannote": package, "pyannote.audio": audio_module}):
                result = run_pyannote_diarization(root / "audio.wav", backend_config=config)

            self.assertEqual(calls["model_path"], str(snapshot))
            self.assertEqual(calls["offline_environment"], {
                "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                "TRANSFORMERS_OFFLINE": "1", "PYANNOTE_METRICS_ENABLED": "0",
            })
            self.assertEqual({name: os.environ.get(name) for name in env_names}, original_environment)
            self.assertEqual(calls["audio_path"], str(root / "audio.wav"))
            self.assertEqual(calls["constraints"], {})
            self.assertEqual(len(result.turns), 2)
            self.assertEqual(len(result.exclusive_turns), 1)
            self.assertTrue(result.turns[0].overlap)
            self.assertIsNone(result.turns[0].confidence)
            self.assertEqual(result.model_digest, self._directory_digest(snapshot))

    def test_community_runtime_rejects_wrong_revision_path_or_digest_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot = root / PYANNOTE_COMMUNITY_REVISION
            snapshot.mkdir()
            (snapshot / "config.yaml").write_text("fixed local config", encoding="utf-8")
            invalid_configs = [
                {"model_id": PYANNOTE_MODEL_ID, "revision": "un-pinned", "snapshot_path": str(snapshot)},
                {"model_id": "other/model", "revision": PYANNOTE_COMMUNITY_REVISION, "snapshot_path": str(snapshot)},
                {"model_id": PYANNOTE_MODEL_ID, "revision": PYANNOTE_COMMUNITY_REVISION,
                 "snapshot_path": str(snapshot)},
                {"model_id": PYANNOTE_MODEL_ID, "revision": PYANNOTE_COMMUNITY_REVISION,
                 "snapshot_path": str(snapshot), "model_digest": "0" * 64},
            ]
            for config in invalid_configs:
                with self.subTest(config=config):
                    with self.assertRaises(DiarizationBackendUnavailable):
                        run_pyannote_diarization(root / "audio.wav", backend_config=config)

    def test_pyannote_status_requires_pinned_local_snapshot_and_hash(self) -> None:
        from workbench.speaker_transcript import detect_speaker_backend_status

        config = {"diarization": {"backend": "pyannote"}}
        with patch("workbench.speaker_transcript.package_available", side_effect=lambda name: name == "pyannote.audio"), \
             patch("workbench.speaker_transcript.find_hf_token", return_value="HF_TOKEN"):
            status = detect_speaker_backend_status(config)
        self.assertFalse(status["pyannote_snapshot_ready"])
        self.assertFalse(status["ready"])
        self.assertFalse(status["backends"]["pyannote-community-1"])

    def test_pyannote_legacy_adapter_only_passes_supported_arguments(self) -> None:
        captured: dict[str, int] = {}

        def fake_pyannote(audio_path: Path, *, min_speakers: int, max_speakers: int):
            captured.update(min_speakers=min_speakers, max_speakers=max_speakers)
            return [DiarizationTurn("SPEAKER_00", 0.0, 1.0)]

        with patch("workbench.speaker_transcript.run_pyannote_diarization", new=fake_pyannote):
            result = PyannoteCommunityBackend().diarize(
                Path("audio.wav"),
                DiarizationOptions(min_speakers=2, max_speakers=3, intro_end=4.0, transcript_segments=[]),
            )
        self.assertEqual(captured, {"min_speakers": 2, "max_speakers": 3})
        self.assertEqual(result.turns[0].raw_label, "SPEAKER_00")

    def test_community_adapter_does_not_invent_speaker_count_bounds(self) -> None:
        captured = {}
        def fake_pyannote(audio_path: Path, **kwargs):
            captured.update(kwargs)
            return [DiarizationTurn("SPEAKER_00", 0.0, 1.0)]
        with patch("workbench.speaker_transcript.run_pyannote_diarization", new=fake_pyannote):
            result = PyannoteCommunityBackend().diarize(Path("audio.wav"), DiarizationOptions())
        self.assertNotIn("min_speakers", captured)
        self.assertNotIn("max_speakers", captured)
        self.assertIn("backend_config", captured)
        self.assertEqual(result.turns[0].raw_label, "SPEAKER_00")

    def test_render_speaker_text_preserves_dialogue_content(self) -> None:
        rendered = render_speaker_text(
            [{"speaker": "孟岩", "start": 0.0, "end": 2.0, "text": "嗯大家好。"}],
            with_timestamps=False,
        )
        self.assertIn("孟岩：嗯大家好。", rendered)

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
            self.assertEqual(payload["speaker_turns"][0]["speaker"], "说话人 1")
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

    def test_embedding_windows_exclude_filler_and_music_only_segments(self) -> None:
        segments = [
            TranscriptSegment(0.0, 3.0, "嗯嗯嗯"),
            TranscriptSegment(3.0, 6.0, "（音乐）"),
            TranscriptSegment(6.0, 9.0, "我们开始讨论背景。"),
        ]
        self.assertFalse(has_lexical_content("嗯，啊啊"))
        self.assertFalse(has_lexical_content("啊啊啊啊啊啊跟"))
        self.assertFalse(has_lexical_content("[background music]"))
        self.assertTrue(has_lexical_content("我们开始讨论背景。"))
        self.assertTrue(has_lexical_content("金额"))
        self.assertTrue(has_lexical_content("哈佛"))
        windows = build_audio_windows(segments, intro_end=0.0, lexical_only=True)
        self.assertEqual([(item.start, item.end) for item in windows], [(6.0, 9.0)])

    def test_overlapping_embedding_windows_become_exclusive_and_keep_gaps(self) -> None:
        turns = exclusive_diarization_turns(
            [
                DiarizationTurn("Speaker 1", 0.0, 3.2),
                DiarizationTurn("Speaker 2", 1.6, 4.8),
                DiarizationTurn("Speaker 1", 8.0, 10.0),
            ]
        )
        self.assertEqual(
            [(item.raw_label, round(item.start, 3), round(item.end, 3)) for item in turns],
            [("Speaker 1", 0.0, 2.4), ("Speaker 2", 2.4, 4.8), ("Speaker 1", 8.0, 10.0)],
        )
        for previous, current in zip(turns, turns[1:]):
            self.assertLessEqual(previous.end, current.start)
        self.assertEqual(turns[1].overlap, False)

    def test_speaker_embedding_clustering_uses_cosine_average(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is an optional speaker embedding dependency")

        calls = []

        class FakeClustering:
            def __init__(self, **kwargs):
                calls.append(kwargs)

            def fit_predict(self, matrix):
                self.matrix = matrix
                return np.asarray([0, 1], dtype=np.int64)

        labels = cluster_speaker_embeddings([[3.0, 0.0], [0.0, 4.0]], 2, FakeClustering, np_module=np)
        self.assertEqual(labels.tolist(), [0, 1])
        self.assertEqual(calls, [{"n_clusters": 2, "metric": "cosine", "linkage": "average"}])

    def test_directory_input_writes_new_speaker_artifacts_to_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source-run"
            output = root / "speaker-run"
            source.mkdir()
            audio = source / "转录音频.wav"
            audio.write_bytes(b"source audio")
            transcript = source / "transcript.json"
            original = json.dumps(
                {"transcription": [{"timestamps": {"from": "00:00:00,000", "to": "00:00:03,000"}, "text": "开始讨论。"}]},
                ensure_ascii=False,
            )
            transcript.write_text(original, encoding="utf-8")
            context = source / "source.context.json"
            context.write_text(json.dumps({"speaker_candidates": [{"name": "孟岩"}]}), encoding="utf-8")
            workspace = resolve_speaker_workspace(source, quality="fast", output_dir=output)
            self.assertEqual(workspace["run_dir"], output.resolve())
            self.assertEqual(workspace["transcript_json_path"], transcript.resolve())
            self.assertEqual(workspace["audio_path"], audio.resolve())

            with patch(
                "workbench.speaker_transcript.ensure_speaker_dependencies",
                return_value={"selected_backend": "pyannote", "fallback_backend_available": True, "ready": True},
            ), patch("workbench.speaker_transcript.enhance_voice_audio", return_value=output / "derived" / "audio.voice_reference.wav"), patch(
                "workbench.speaker_transcript.run_diarization",
                return_value=("pyannote", [DiarizationTurn("Speaker 1", 0.0, 3.0)]),
            ):
                payload = build_speaker_transcript(source, output_dir=output)

            self.assertEqual(Path(payload["run_dir"]), output.resolve())
            self.assertTrue((output / "transcript.speakers.json").exists())
            self.assertTrue((output / "diarization.raw.json").exists())
            self.assertFalse((source / "transcript.speakers.json").exists())
            self.assertEqual(transcript.read_text(encoding="utf-8"), original)

    def test_legacy_voice_enrollment_dict_cannot_name_a_speaker(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = root / "转录音频.wav"
            audio.write_bytes(b"source audio")
            transcript = root / "transcript.json"
            transcript.write_text(
                json.dumps(
                    {"transcription": [{"timestamps": {"from": "00:00:00,000", "to": "00:00:03,000"}, "text": "开始讨论。"}]},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            evidence = [
                {
                    "raw_speaker_label": "Speaker 1",
                    "display_name": "孟岩",
                    "person_id": "person_mengyan",
                    "evidence_type": "voice_enrollment",
                    "source": "voice_enrollment",
                    "confidence": 0.96,
                    "threshold": 0.78,
                }
            ]
            config = {"speaker_identity": {"enable_voice_enrollment": True, "minimum_confidence": 0.90}}
            with patch(
                "workbench.speaker_transcript.ensure_speaker_dependencies",
                return_value={"selected_backend": "pyannote", "fallback_backend_available": True, "ready": True},
            ), patch("workbench.speaker_transcript.enhance_voice_audio", return_value=root / "derived" / "audio.voice_reference.wav"), patch(
                "workbench.speaker_transcript.run_diarization",
                return_value=("pyannote", [DiarizationTurn("Speaker 1", 0.0, 3.0)]),
            ), patch(
                "workbench.speaker_transcript.collect_voice_enrollment_evidence",
                return_value=(evidence, {"enabled": True, "match_count": 0, "reason": "disabled_by_recording_only_validation_scope"}),
            ):
                payload = build_speaker_transcript(root, config=config)

            self.assertEqual(payload["speakers"][0]["display_name"], "unknown")
            self.assertEqual(payload["speakers"][0]["identity_status"], "unknown")
            self.assertEqual(payload["voice_enrollment"]["match_count"], 0)

    def test_text_only_context_helper_cannot_identify_any_voice(self) -> None:
        turns = [
            # Text rows have no independently verified acoustic provenance.
            SpeakerTextTurn("Speaker 1", 0.0, 2.0, "大家好，我是李继刚。"),
            SpeakerTextTurn("Speaker 2", 2.0, 4.0, "我们开始聊聊。"),
        ]
        renamed, mapping, inference = apply_context_speaker_names(
            turns,
            context={"speaker_candidates": [{"name": "孟岩"}, {"name": "李继刚"}]},
        )
        self.assertEqual(mapping["Speaker 1"], "")
        self.assertEqual(mapping["Speaker 2"], "")
        self.assertEqual([turn.speaker for turn in renamed], ["Speaker 1", "Speaker 2"])
        self.assertEqual(inference["status"], "unresolved")

    def test_text_self_introduction_never_changes_acoustic_switch(self) -> None:
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
        self.assertEqual([turn.speaker for turn in turns], ["Speaker 2"])
        self.assertEqual(turns[0].text, "".join(s.text for s in segments))

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
        # Alignment retains anonymous acoustic labels until identity resolution.
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

    def test_word_alignment_preserves_zero_duration_tokens_and_short_gaps(self) -> None:
        segment = TranscriptSegment(
            0.0,
            4.0,
            "甲乙丙丁",
            "seg_zero",
            words=[
                TranscriptWord("甲", 0.5, 0.5),
                TranscriptWord("乙", 1.0, 1.4),
                TranscriptWord("丙", 2.0, 2.0),
                TranscriptWord("丁", 2.6, 3.0),
            ],
        )
        aligned, unassigned = align_segments_to_speakers(
            [segment],
            [DiarizationTurn("SPEAKER_00", 0.0, 1.8), DiarizationTurn("SPEAKER_01", 2.2, 4.0)],
            0.0,
        )
        self.assertEqual(unassigned, [])
        self.assertEqual("".join(item.text for item in aligned), segment.text)
        self.assertEqual([item.speaker for item in aligned], ["归属待核对", "SPEAKER_00", "归属待核对", "SPEAKER_01"])
        self.assertEqual(aligned[0].alignment_spans[0]["assignment_reason"], "missing_duration")
        self.assertEqual(aligned[2].alignment_spans[0]["assignment_reason"], "missing_duration")

    def test_segment_without_word_timestamps_is_not_cut_mid_sentence(self) -> None:
        segment = TranscriptSegment(0.0, 4.0, "这是一句完整的话。", "seg_coarse")
        aligned, unassigned = align_segments_to_speakers(
            [segment],
            [DiarizationTurn("SPEAKER_00", 0.0, 3.0), DiarizationTurn("SPEAKER_01", 3.0, 4.0)],
            0.0,
        )
        self.assertEqual(unassigned, [])
        self.assertEqual([(item.speaker, item.text) for item in aligned], [("归属待核对", segment.text)])

    def test_word_crossing_exclusive_boundary_is_uncertain_without_reallocation(self) -> None:
        segment = TranscriptSegment(
            0.0,
            2.0,
            "投资",
            "seg_boundary",
            words=[TranscriptWord("投", 0.9, 1.1), TranscriptWord("资", 1.1, 1.4)],
        )
        aligned, _ = align_segments_to_speakers(
            [segment],
            [DiarizationTurn("SPEAKER_00", 0.0, 1.0), DiarizationTurn("SPEAKER_01", 1.0, 2.0)],
            0.0,
        )
        self.assertEqual([(r.speaker, r.text) for r in aligned], [("归属待核对", "投"), ("SPEAKER_01", "资")])
        self.assertEqual("".join(item.text for item in aligned), "投资")

    def test_self_introduction_does_not_move_greeting_or_invent_times(self) -> None:
        text = "简单先介绍一下自己呃大家好我是李继刚"
        words = [TranscriptWord(char, index * 0.1, (index + 1) * 0.1) for index, char in enumerate(text)]
        boundary = text.index("我是") * 0.1
        aligned, _ = align_segments_to_speakers(
            [TranscriptSegment(0.0, len(text) * 0.1, text, "seg_intro", words=words)],
            [
                DiarizationTurn("SPEAKER_00", 0.0, boundary),
                DiarizationTurn("SPEAKER_01", boundary, len(text) * 0.1),
            ],
            0.0,
            context={"speaker_candidates": ["孟岩", "李继刚"]},
        )
        self.assertEqual("".join(item.text for item in aligned), text)
        self.assertEqual(aligned[0].text, "简单先介绍一下自己呃大家好")
        self.assertEqual(aligned[1].text, "我是李继刚")
        self.assertEqual(aligned[1].speaker, "SPEAKER_01")
        self.assertEqual(aligned[1].start, boundary)
        self.assertEqual(aligned[0].end, boundary)

    def test_two_person_metadata_does_not_identify_remaining_host(self) -> None:
        segments = [
            TranscriptSegment(0.0, 2.0, "欢迎来到节目。", "seg_host"),
            TranscriptSegment(2.0, 4.0, "大家好，我是李继刚。", "seg_guest"),
        ]
        turns = [DiarizationTurn("SPEAKER_00", 0.0, 2.0), DiarizationTurn("SPEAKER_01", 2.0, 4.0)]
        identities = DefaultSpeakerIdentityResolver().resolve(
            turns,
            segments,
            context={
                "speaker_candidates": [
                    {"name": "孟岩", "role": "host", "source": "podcast.author"},
                    {"name": "李继刚", "role": "guest", "source": "shownotes"},
                ]
            },
        )
        by_label = identities.by_raw_label()
        self.assertEqual(by_label["SPEAKER_01"].display_name, "李继刚")
        self.assertEqual(by_label["SPEAKER_00"].display_name, "unknown")
        self.assertEqual(by_label["SPEAKER_00"].identity_status, "unknown")
        self.assertEqual(by_label["SPEAKER_00"].identity_evidence, [])

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

    def test_sherpa_1137_adapter_builds_official_nested_config_and_sorts_result(self) -> None:
        class FakePyannoteConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeSegmentationConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeEmbeddingConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeClusteringConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeDiarizationConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def validate(self):
                return True

        class FakeSegment:
            def __init__(self, start, end, speaker):
                self.start = start
                self.end = end
                self.speaker = speaker

        class FakeResult:
            def __init__(self):
                self.sorted = False

            def sort_by_start_time(self):
                self.sorted = True
                return [FakeSegment(0.0, 0.5, 1), FakeSegment(0.5, 1.0, 0)]

        captured = {}

        class FakeDiarizer:
            sample_rate = 16000

            def __init__(self, config):
                captured["config"] = config

            def process(self, samples):
                captured["samples"] = samples
                return FakeResult()

        class FakeSherpa:
            OfflineSpeakerDiarization = FakeDiarizer
            OfflineSpeakerDiarizationConfig = FakeDiarizationConfig
            OfflineSpeakerSegmentationModelConfig = FakeSegmentationConfig
            OfflineSpeakerSegmentationPyannoteModelConfig = FakePyannoteConfig
            SpeakerEmbeddingExtractorConfig = FakeEmbeddingConfig
            FastClusteringConfig = FakeClusteringConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = root / "audio.wav"
            with wave.open(str(audio), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\x00\x00" * 16000)
            segmentation = root / "segmentation.onnx"
            embedding = root / "embedding.onnx"
            segmentation.write_bytes(b"model")
            embedding.write_bytes(b"model")
            with patch("workbench.diarization.importlib.import_module", return_value=FakeSherpa()):
                result = SherpaOnnxBackend().diarize(
                    audio,
                    options={
                        "min_speakers": 2,
                        "backend_config": {"model_paths": [str(segmentation), str(embedding)]},
                    },
                )

        config = captured["config"].kwargs
        self.assertEqual(config["segmentation"].kwargs["pyannote"].kwargs["model"], str(segmentation))
        self.assertEqual(config["embedding"].kwargs["model"], str(embedding))
        self.assertEqual(config["clustering"].kwargs, {"num_clusters": -1, "threshold": 0.5})
        self.assertEqual(config["min_duration_on"], 0.3)
        self.assertEqual(config["min_duration_off"], 0.5)
        samples = captured["samples"]
        self.assertEqual(len(samples), 16000)
        self.assertEqual(result.turns[0].raw_label, "SPEAKER_01")
        self.assertEqual(result.turns[1].raw_label, "SPEAKER_00")

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
