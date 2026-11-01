"""Whisper跨缝反例、整cue字幕、原生证据及真实publisher；模型边界为替身。"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from workbench.contracts import PipelineError
from workbench.config import NextEchoConfig
from workbench.io_utils import file_hash, write_json
from workbench.publication import publish_pipeline, verify_publication
from workbench.transcription import (
    _append_shifted_srt, _append_shifted_vtt, _load_shifted_transcription_segments,
    _whisper_chunk_info, transcribe_audio_file_in_chunks,
    transcribe_audio_with_fallback, transcribe_single_source, transcribe_audio_file,
    LONG_AUDIO_CPU_FALLBACK_MS,
)
from workbench.whisper_chunks import shifted_segments, subtitle_cues
from workbench.sources import resolve_source


def segment(start, end, text):
    return {"offsets": {"from": start, "to": end}, "text": text}


class WhisperChunkTests(unittest.TestCase):
    def fixture(self, root, formats=None, audio=None):
        root.mkdir(parents=True, exist_ok=True)
        if audio is None:
            audio = root / "audio.wav"
            audio.write_bytes(b"synthetic source; not real audio")
        calls = []
        def extract(**kwargs):
            kwargs["target_path"].write_bytes(b"synthetic chunk")
        def recognize(**kwargs):
            calls.append(kwargs)
            out = kwargs["output_dir"] / kwargs["output_base"]
            if len(calls) == 1:
                rows = [segment(1000, 2000, "开头"), segment(598000, 600000, "非常非常重要")]
            else:
                rows = [segment(28000, 29900, "只在前片的上下文"),
                        segment(29000, 35000, "非常非常重要，李继刚的专名后缀不能漏"),
                        segment(35000, 36000, "最后一句")]
            self.assertIn("json", kwargs["output_formats"])
            write_json(out.with_suffix(".json"), {"transcription": rows, "native_extra": "retained"})
            out.with_suffix(".vtt").write_text("WEBVTT\n\n00:00.000 --> 00:01.000\nWRONG_RAW_CUE\n")
        with patch("workbench.transcription.extract_audio_segment", side_effect=extract), patch("workbench.transcription.run_whisper_cli_cpu", side_effect=recognize):
            transcribe_audio_file_in_chunks(audio_path=audio, output_dir=root, output_base="transcript",
                whisper_bin="fixture", model_path=root / "model.bin", language="zh",
                output_formats=formats or ["txt", "json", "srt", "vtt"], duration_ms=606000)
        return calls

    def test_crossing_native_sentence_keeps_full_suffix_and_measured_range(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "chunk.json"
            write_json(path, {"transcription": [segment(28000, 30000, "旧上下文"),
                segment(29000, 35000, "599到605秒的专名和后缀"), segment(35000, 36000, "后句")]})
            d = _load_shifted_transcription_segments(path, 570000, trim_before_ms=600000)
        self.assertEqual([r["text"] for r in d], ["599到605秒的专名和后缀", "后句"])
        self.assertEqual(d[0]["offsets"], {"from": 599000, "to": 605000})
        self.assertEqual(d[0]["source_offsets"], {"from": 29000, "to": 35000})
        self.assertEqual(d[0]["seam_review"]["policy"], "retain_whole_native_segment")
        self.assertNotIn("seam_review", d[1])

    def test_whole_srt_vtt_cues_drop_context_without_orphan_body(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            srt = root / "chunk.srt"
            srt.write_text("1\r\n00:00:28,000 --> 00:00:29,900\r\nSHOULD_DROP\r\n\r\n2\r\n00:00:29,000 --> 00:00:35,000\r\n专名后缀\r\n多行\r\n")
            vtt = root / "chunk.vtt"
            vtt.write_text("WEBVTT\n\nNOTE developer note\nnot a cue\n\nold-id\n00:28.000 --> 00:29.900\nSHOULD_DROP\n\nnew-id\n00:29.000 --> 00:35.000 align:start position:10%\n专名后缀\n多行\n")
            a, b = [], ["WEBVTT", ""]
            self.assertEqual(_append_shifted_srt(srt, a, 570000, 1, trim_before_ms=600000), 2)
            _append_shifted_vtt(vtt, b, 570000, trim_before_ms=600000)
            self.assertNotIn("SHOULD_DROP", "\n".join(a + b))
            self.assertNotIn("old-id", b)
            self.assertIn("new-id", b)
            self.assertIn("00:09:59.000 --> 00:10:05.000 align:start position:10%", b)
            for text, vtt_mode in (("\n".join(a), False), ("\n".join(b), True)):
                cue = subtitle_cues(text, vtt=vtt_mode)
                self.assertEqual(len(cue), 1)
                self.assertEqual(cue[0]["lines"], ["专名后缀", "多行"])
                self.assertEqual((cue[0]["start_ms"], cue[0]["end_ms"]), (599000, 605000))

    def test_invalid_native_intervals_or_incomplete_cue_do_not_invent_times(self):
        for row in (segment(0, 0, "字"), segment(-1, 2, "字"), segment(2, 1, "字"), {"text": "无时间"}):
            with self.assertRaises(PipelineError):
                shifted_segments({"transcription": [row]}, 0)
        with self.assertRaises(PipelineError):
            subtitle_cues("WEBVTT\n\n孤儿正文", vtt=True)
        self.assertEqual(shifted_segments({"transcription": [segment(0, 0, " ")]}, 0), [])

    def test_txt_json_srt_vtt_share_retained_native_segments_and_keep_repetition(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            calls = self.fixture(root)
            self.assertEqual(len(calls), 2)
            d = json.loads((root / "transcript.json").read_text())
            rows = d["transcription"]
            expected = ["开头", "非常非常重要", "非常非常重要，李继刚的专名后缀不能漏", "最后一句"]
            self.assertEqual([r["text"] for r in rows], expected)
            self.assertEqual(rows[2]["offsets"], {"from": 599000, "to": 605000})
            self.assertTrue(d["review_required"])
            for suffix, vtt in (("srt", False), ("vtt", True)):
                cues = subtitle_cues((root / f"transcript.{suffix}").read_text(), vtt=vtt)
                self.assertEqual([c["lines"][0] for c in cues], expected)
                self.assertEqual([(c["start_ms"], c["end_ms"]) for c in cues], [(1000, 2000), (598000, 600000), (599000, 605000), (605000, 606000)])
            self.assertNotIn("WRONG_RAW_CUE", (root / "transcript.vtt").read_text())
            info = _whisper_chunk_info(root, "transcript")
            self.assertTrue(info["review_required"])
            for record in info["chunk_sources"]:
                path = Path(record["json_path"])
                self.assertEqual(file_hash(path), record["sha256"])
                self.assertEqual(json.loads(path.read_text())["native_extra"], "retained")

    def test_text_only_still_requests_native_json_and_keeps_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root, ["txt"])
            self.assertFalse((root / "transcript.json").exists())
            self.assertTrue(_whisper_chunk_info(root, "transcript")["review_required"])
            self.assertIn("专名后缀不能漏", (root / "transcript.txt").read_text())

    def test_raw_chunk_hash_and_path_validation_block_tamper_or_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            info = _whisper_chunk_info(root, "transcript")
            raw = Path(info["chunk_sources"][0]["json_path"])
            raw.write_text("tampered")
            with self.assertRaises(PipelineError) as exc:
                _whisper_chunk_info(root, "transcript")
            self.assertEqual(exc.exception.code, "CACHE_MISMATCH")
            audit = root / "transcript.chunks.audit.json"
            data = json.loads(audit.read_text())
            data["chunks"][0]["relative_json"] = "../outside.json"
            write_json(audit, data)
            with self.assertRaises(PipelineError):
                _whisper_chunk_info(root, "transcript")

    def test_chunk_audit_is_bound_to_normalized_audio_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            audio = root / "audio.wav"
            self.assertTrue(_whisper_chunk_info(root, "transcript", audio_path=audio)["review_required"])
            audio.write_bytes(b"different audio")
            with self.assertRaises(PipelineError) as exc:
                _whisper_chunk_info(root, "transcript", audio_path=audio)
            self.assertEqual(exc.exception.code, "CACHE_MISMATCH")

    def test_non_json_native_artifacts_are_retained_and_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            info = _whisper_chunk_info(root, "transcript")
            native_vtt = next(Path(r["artifact_path"]) for r in info["chunk_evidence"] if r["artifact_path"].endswith(".vtt"))
            self.assertIn("WRONG_RAW_CUE", native_vtt.read_text())
            native_vtt.write_text("changed evidence")
            with self.assertRaises(PipelineError) as exc:
                _whisper_chunk_info(root, "transcript")
            self.assertEqual(exc.exception.code, "CACHE_MISMATCH")

    def test_cache_relocation_rebinds_current_run_artifacts_without_changing_native_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            self.fixture(root / "parent")
            shutil.copytree(root / "parent", root / "child")
            a = _whisper_chunk_info(root / "parent", "transcript")
            b = _whisper_chunk_info(root / "child", "transcript")
            self.assertEqual([r["sha256"] for r in a["chunk_sources"]], [r["sha256"] for r in b["chunk_sources"]])
            self.assertTrue(all(Path(r["json_path"]).is_relative_to(root / "child") for r in b["chunk_sources"]))

    def test_seam_review_publication_is_partial_and_hashes_all_native_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            item = run / "items/one"
            self.fixture(item)
            info = {"actual_engine": "whisper-cpp", **_whisper_chunk_info(item, "transcript")}
            result = {"status": "ready", "title": "片段", "output_dir": str(item),
                "text_path": str(item / "transcript.txt"), "json_path": str(item / "transcript.json"), "asr": info}
            receipt = publish_pipeline(run, job_id="chunk-review", results=[result], plan={})
            self.assertEqual(receipt["status"], "partial")
            self.assertNotIn("识别分片交界待审", Path(receipt["primary_file"]).read_text())
            self.assertTrue(receipt["items"][0]["asr"]["review_required"])
            for r in info["chunk_sources"]:
                self.assertIn(r["json_path"], receipt["artifact_hashes"])
            self.assertEqual(verify_publication(run, job_id="chunk-review")["status"], "partial")
            Path(info["chunk_sources"][0]["json_path"]).write_text("tampered")
            with self.assertRaises(PipelineError):
                verify_publication(run, job_id="chunk-review")

    def test_protected_pipeline_failures_do_not_trigger_legacy_model_retries(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = dict(audio_path=root / "audio.wav", output_dir=root, output_base="transcript",
                whisper_bin="fixture", whisper_model_path=root / "model.bin", whisper_model_name="large-v3",
                language="zh", output_formats=["txt", "json"], model="fixture", config=NextEchoConfig({}))
            for code in ("CANCELLED", "RESOURCE_LIMIT", "RESOURCE_BLOCKED", "STAGE_TIMEOUT", "PROTOCOL_ERROR", "CACHE_MISMATCH"):
                with self.subTest(code=code), patch("workbench.transcription.validate_whisper_model", return_value={"bytes": 100}), patch("workbench.transcription.transcribe_audio_file", side_effect=PipelineError(code, "fixture")) as invoke, patch("workbench.transcription.resolve_whisper_model") as resolve:
                    with self.assertRaises(PipelineError) as exc:
                        transcribe_audio_with_fallback(engine="whisper-cpp", **args)
                    self.assertEqual(exc.exception.code, code)
                    self.assertEqual(invoke.call_count, 1)
                    resolve.assert_not_called()
            with patch("workbench.transcription.transcribe_qwen_mlx", side_effect=PipelineError("CANCELLED", "fixture")) as invoke, patch("workbench.transcription.resolve_whisper_model") as resolve:
                with self.assertRaises(PipelineError):
                    transcribe_audio_with_fallback(engine="qwen3-asr", **args)
                self.assertEqual(invoke.call_count, 1)
                resolve.assert_not_called()

    def test_long_audio_preserves_requested_auto_or_explicit_language(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for language in ("auto", "en", "zh"):
                with self.subTest(language=language), patch("workbench.transcription.probe_audio_duration_ms", return_value=LONG_AUDIO_CPU_FALLBACK_MS), patch("workbench.transcription.transcribe_audio_file_in_chunks") as chunks:
                    transcribe_audio_file(audio_path=root / "audio.wav", output_dir=root, output_base="transcript",
                        whisper_bin="fixture", model_path=root / "model.bin", language=language, output_formats=["txt"])
                    self.assertEqual(chunks.call_args.kwargs["language"], language)

    def test_asr_cache_checks_raw_chunks_and_timeline_implementation(self):
        self.check_asr_cache("whisper-cpp")

    def test_legacy_qwen_to_whisper_cache_checks_same_timeline_implementation(self):
        self.check_asr_cache("qwen3-asr")

    def check_asr_cache(self, requested_engine):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            source = root / "source.wav"
            source.write_bytes(b"source fixture")
            model = root / "ggml-base.bin"
            model.write_bytes(b"synthetic model identity")
            calls = []
            def normalize(**kwargs):
                kwargs["target_path"].write_bytes(b"normalized fixture")
            def recognize(**kwargs):
                calls.append(kwargs)
                self.fixture(kwargs["output_dir"], audio=kwargs["audio_path"])
                return {"actual_engine": "whisper-cpp", **_whisper_chunk_info(kwargs["output_dir"], "transcript", audio_path=kwargs["audio_path"])}
            req = dict(descriptor=resolve_source(str(source)), items_dir=root / "old/items", model_path=model,
                whisper_bin="fixture-whisper", ffmpeg_bin="fixture-ffmpeg", language="zh", max_seconds=0,
                output_formats=["txt", "json", "srt", "vtt"], glossary_text="", enable_enhancement=False,
                ai_polish=False, ai_style="", index=1, asr_engine=requested_engine, asr_model="fixture",
                defer_enhancement=True, config=NextEchoConfig({"runtime": {"resume_verified": True}}))
            with patch("workbench.transcription.extract_audio", side_effect=normalize), patch("workbench.transcription.transcribe_audio_with_fallback", side_effect=recognize):
                first = transcribe_single_source(**req)
                self.assertEqual(transcribe_single_source(**req)["status"], "cached")
                self.assertEqual(len(calls), 1)
                raw = Path(first["asr"]["chunk_sources"][0]["json_path"])
                original = raw.read_bytes()
                raw.write_text("corruption")
                with self.assertRaises(PipelineError) as exc:
                    transcribe_single_source(**req)
                self.assertEqual(exc.exception.code, "CACHE_MISMATCH")
                self.assertEqual(len(calls), 1)
                raw.write_bytes(original)
                parent_sha = file_hash(Path(first["json_path"]))
                shutil.copytree(root / "old/items", root / "new/items")
                req["items_dir"] = root / "new/items"
                def changed(path):
                    return "f" * 64 if Path(path).name == "whisper_chunks.py" else file_hash(Path(path))
                with patch("workbench.transcription.file_hash", side_effect=changed):
                    second = transcribe_single_source(**req)
                self.assertEqual(second["status"], "ready")
                self.assertEqual(len(calls), 2)
                self.assertEqual(file_hash(Path(first["json_path"])), parent_sha)


if __name__ == "__main__":
    unittest.main()
