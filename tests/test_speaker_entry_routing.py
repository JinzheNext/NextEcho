"""Real entry orchestration and artifacts; synthetic data only at model boundaries."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import app as web
from workbench import cli
from workbench.acoustic_dialogue import build_dialogue_payload, write_dialogue_outputs
from workbench.config import NextEchoConfig
from workbench.contracts import PipelineError, PipelineRequest, plan_request
from workbench.io_utils import file_hash, write_json
from workbench.pipeline import execute_plan
from workbench.postprocess import load_segments
from workbench.publication import verify_publication
from workbench.speaker_artifacts import read_speaker_file, write_v3_outputs
from workbench.speaker_service import build_run_speakers
from workbench.transcription import format_transcript_attribution
from tests.test_publication import factory_item


def asr_fixture(root: Path, *, engines=("fun-asr-nano",), partial_second=False):
    results = []
    for index, engine in enumerate(engines):
        name = f"item-{index + 1}"
        item = factory_item(root, name)
        folder = Path(item["output_dir"])
        text = "金额36万元。大家好，我是李继刚。"
        write_json(
            folder / "transcript.json",
            {
                "transcription": [
                    {
                        "id": "raw-1",
                        "text": text,
                        "offsets": {"from": 0, "to": len(text) * 200},
                    }
                ]
            },
        )
        (folder / "transcript.txt").write_text(format_transcript_attribution(text))
        item.update(
            char_count=len(text),
            asr={
                "actual_engine": engine,
                "timing_precision": "vad-segment"
                if engine in {"fun-asr-nano", "sensevoice"}
                else "segment",
            },
        )
        if partial_second and index == 1:
            item["test_unanchored"] = True
        write_json(folder / "metadata.json", item)
        results.append(item)
    manifest = root / "manifest.json"
    write_json(manifest, {"results": results, "completed_at": "2026-10-20T00:00:00Z"})
    return {
        "output_dir": root,
        "manifest_path": manifest,
        "errors_path": None,
        "results": results,
        "model_name": "synthetic-model",
        "model_path": "",
        "run_config_path": root / "run_config.json",
        "elapsed_seconds": 1,
        "asr_engine": engines[0],
    }


def native_measurement_fixture(folder, *, config, progress_callback=None, **kwargs):
    folder = Path(folder)
    raw = [
        vars(segment)
        for segment in load_segments(
            json.loads((folder / "transcript.json").read_text())
        )
    ]
    text = raw[0]["text"]
    boundary = text.index("大家好") * 200 / 1000
    item = json.loads((folder / "metadata.json").read_text())
    tokens = (
        []
        if item.get("test_unanchored")
        else [
            {"text": char, "start_ms": i * 200, "end_ms": None}
            for i, char in enumerate(text)
        ]
    )
    measured = [
        {"speaker": "A", "start": 0, "end": boundary},
        {"speaker": "B", "start": boundary, "end": len(text) * 200 / 1000},
    ]
    payload = build_dialogue_payload(
        raw, raw, tokens, measured, {"speaker_candidates": [{"name": "李继刚"}]}
    )
    payload["raw_sha256"] = file_hash(folder / "transcript.json")
    paths = write_dialogue_outputs(folder, payload)
    if progress_callback:
        progress_callback({"stage": "speaker", "operation": "synthetic-measurement"})
    return {**payload, "paths": paths}


def legacy_measurement_fixture(stage, **kwargs):
    stage = Path(stage)
    raw = json.loads((stage / "transcript.json").read_text())["transcription"][0]
    payload = {
        "schema_version": 3,
        "status": "partial",
        "speakers": [],
        "speaker_map": {},
        "speaker_turns": [
            {
                "speaker": "归属待核对",
                "speaker_id": "",
                "raw_speaker": "",
                "text": raw["text"],
                "start": 0,
                "end": raw["offsets"]["to"] / 1000,
                "segment_ids": [raw["id"]],
            }
        ],
    }
    write_v3_outputs(stage, payload)
    write_json(
        stage / "diarization.raw.json",
        {"annotation_kind": "acoustic-diarization", "turns": []},
    )
    (stage / "diarization.raw.rttm").write_text("")
    return payload


class SpeakerEntryRoutingTests(unittest.TestCase):
    def test_legacy_cli_actual_command_routes_all_coarse_items_and_keeps_partial(self):
        with (
            tempfile.TemporaryDirectory() as d,
            patch.dict(os.environ, {"NEXTECHO_RUNTIME_DIR": str(Path(d) / "runtime")}),
        ):
            root = Path(d) / "cli"
            original = {}

            def asr(*args, output_dir, **kwargs):
                payload = asr_fixture(
                    output_dir,
                    engines=("fun-asr-nano", "sensevoice"),
                    partial_second=True,
                )
                original.update(
                    {
                        Path(item["json_path"]): file_hash(Path(item["json_path"]))
                        for item in payload["results"]
                    }
                )
                return payload

            with (
                patch(
                    "workbench.transcription._transcribe_media_sources", side_effect=asr
                ),
                patch(
                    "workbench.acoustic_dialogue.build_acoustic_dialogue",
                    side_effect=native_measurement_fixture,
                ) as measured,
                patch(
                    "workbench.cli.run_doctor",
                    return_value=SimpleNamespace(memory_gb=64),
                ),
                patch("workbench.cli.resolve_sources", return_value=[]),
                patch(
                    "workbench.cli.build_speaker_transcript",
                    side_effect=AssertionError(
                        "coarse ASR must use native measurements"
                    ),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                code = cli.main(
                    [
                        "transcribe",
                        "https://example.org/synthetic",
                        "--speaker-mode",
                        "required",
                        "--postprocess-profile",
                        "off",
                        "--enhance",
                        "--output-dir",
                        str(root),
                        "--json",
                    ]
                )
            self.assertEqual(code, 0)  # Legacy exit code remains compatible.
            self.assertEqual(measured.call_count, 2)
            receipt = verify_publication(root)
            self.assertEqual(receipt["status"], "partial")
            body = Path(receipt["primary_file"]).read_text()
            self.assertIn("李继刚：", body)
            self.assertIn("归属待核对：", body)
            self.assertTrue(
                all(file_hash(path) == digest for path, digest in original.items())
            )
            first = root / "items/item-1"
            enhanced = json.loads((first / "transcript.enhanced.json").read_text())
            speakers = read_speaker_file(first / "transcript.speakers.json")
            self.assertEqual(
                [r["id"] for r in enhanced["enhanced_transcription"]],
                [r["id"] for r in speakers["speaker_turns"]],
            )
            self.assertFalse(list(first.glob(".speaker-staging-*")))

    def test_web_worker_uses_native_writer_before_actual_deterministic_enhancement_and_publishes_last(
        self,
    ):
        with (
            tempfile.TemporaryDirectory() as d,
            patch.dict(os.environ, {"NEXTECHO_RUNTIME_DIR": str(Path(d) / "runtime")}),
            patch("app.RUNS_ROOT", Path(d)),
        ):
            root = Path(d) / "web"
            seen = []

            def asr(*args, output_dir, **kwargs):
                self.assertIsNone(verify_publication(output_dir))
                seen.append("asr")
                return asr_fixture(output_dir)

            def measured(folder, **kwargs):
                self.assertIsNone(verify_publication(root))
                seen.append("speaker")
                return native_measurement_fixture(folder, **kwargs)

            web.JOBS.clear()
            web.JOBS["routing-web"] = web._new_job(
                "routing-web", web.TRANSCRIPTION_JOB, run_id="web"
            )
            with (
                patch(
                    "workbench.transcription._transcribe_media_sources", side_effect=asr
                ),
                patch(
                    "workbench.acoustic_dialogue.build_acoustic_dialogue",
                    side_effect=measured,
                ),
                patch(
                    "app.build_speaker_transcript",
                    side_effect=AssertionError("wrong speaker route"),
                ),
            ):
                web.run_transcription_job(
                    "routing-web",
                    [],
                    root,
                    "base",
                    "zh",
                    0,
                    "",
                    False,
                    "",
                    asr_engine="fun-asr-nano",
                    speaker_mode="required",
                    postprocess_profile="off",
                )
            self.assertEqual(
                web.JOBS["routing-web"]["status"], "ready", web.JOBS["routing-web"]
            )
            self.assertEqual(seen, ["asr", "speaker"])
            receipt = verify_publication(root)
            self.assertIn("李继刚：", Path(receipt["primary_file"]).read_text())
            item = receipt["items"][0]
            self.assertEqual(item["speaker_status"], "ready")
            self.assertTrue(item["speaker_json_path"] in receipt["artifact_hashes"])
            self.assertTrue(
                item["aggregate_speaker_json_path"] in receipt["artifact_hashes"]
            )
            self.assertEqual(
                web.app.test_client()
                .get("/api/runs/web")
                .get_json()["speaker_transcript"]["schema_name"],
                "nextecho.speaker-aggregate",
            )

    def test_headless_plan_uses_same_coarse_writer_and_publishes_without_polish(self):
        with (
            tempfile.TemporaryDirectory() as d,
            patch.dict(os.environ, {"NEXTECHO_RUNTIME_DIR": str(Path(d) / "runtime")}),
        ):
            root = Path(d) / "headless"
            source = Path(d) / "input.wav"
            source.write_bytes(b"synthetic input")
            config = NextEchoConfig({"asr": {"engine": "sensevoice"}})
            plan = plan_request(
                PipelineRequest(
                    (str(source),), postprocess="off", speaker_mode="required"
                ),
                config,
            )

            def asr(*args, output_dir, **kwargs):
                self.assertTrue(kwargs["defer_delivery"])
                return asr_fixture(output_dir, engines=("sensevoice",))

            with (
                patch("workbench.pipeline.transcribe_media_sources", side_effect=asr),
                patch(
                    "workbench.acoustic_dialogue.build_acoustic_dialogue",
                    side_effect=native_measurement_fixture,
                ) as measured,
            ):
                receipt = execute_plan(plan, root, job_id="native-routing")
            self.assertEqual(measured.call_count, 1)
            self.assertEqual(verify_publication(root), json.loads(json.dumps(receipt)))
            self.assertIn("李继刚：", Path(receipt["primary_file"]).read_text())

    def test_legacy_backend_isolates_original_from_enhanced_and_registers_real_partial(
        self,
    ):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            payload = asr_fixture(root, engines=("whisper-cpp",))
            folder = Path(payload["results"][0]["output_dir"])
            adopted = folder / "transcript.enhanced.json"
            write_json(adopted, {"enhanced_transcription": [{"text": "不是原文"}]})
            protected = file_hash(adopted)

            def legacy(stage, **kwargs):
                self.assertEqual(
                    list(stage.glob("transcript*.json")), [stage / "transcript.json"]
                )
                self.assertNotIn("不是原文", (stage / "transcript.json").read_text())
                return legacy_measurement_fixture(stage, **kwargs)

            result = build_run_speakers(root, mode="required", legacy_builder=legacy)
            item = json.loads((root / "manifest.json").read_text())["results"][0]
            self.assertEqual(result["status"], "partial")
            self.assertEqual(item["speaker_status"], "partial")
            self.assertEqual(file_hash(adopted), protected)
            self.assertTrue(Path(item["diarization_raw_json_path"]).is_file())
            self.assertFalse(list(folder.glob(".speaker-staging-*")))
            self.assertNotIn(
                "归属待核对", [s["display_name"] for s in result["speakers"]]
            )

    def test_resource_failure_propagates_in_auto_and_internal_typeerror_is_not_retried(
        self,
    ):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            asr_fixture(root)
            with patch(
                "workbench.acoustic_dialogue.build_acoustic_dialogue",
                side_effect=PipelineError("RESOURCE_LIMIT", "stop", blocked=True),
            ):
                with self.assertRaises(PipelineError):
                    cli.maybe_build_speaker_transcript(
                        root, mode="auto", min_speakers=2, max_speakers=6
                    )
            with patch(
                "workbench.cli.build_speaker_transcript",
                side_effect=TypeError("inside backend"),
            ) as model:
                with self.assertRaises(TypeError):
                    cli._call_speaker_builder(
                        root, backend="auto", config=NextEchoConfig({})
                    )
                self.assertEqual(model.call_count, 1)

    def test_off_does_not_call_either_backend(self):
        with tempfile.TemporaryDirectory() as d:
            with (
                patch(
                    "workbench.acoustic_dialogue.build_acoustic_dialogue",
                    side_effect=AssertionError("off"),
                ),
                patch(
                    "workbench.cli.build_speaker_transcript",
                    side_effect=AssertionError("off"),
                ),
            ):
                self.assertIsNone(
                    cli.maybe_build_speaker_transcript(
                        Path(d), mode="off", min_speakers=2, max_speakers=6
                    )
                )

    def test_explicit_configured_backend_is_not_silently_replaced_by_native(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            asr_fixture(root)
            config = NextEchoConfig({"diarization": {"backend": "segment-clustering"}})
            seen = []

            def legacy(stage, **kwargs):
                seen.append(kwargs["backend"])
                return legacy_measurement_fixture(stage, **kwargs)

            with patch(
                "workbench.acoustic_dialogue.build_acoustic_dialogue",
                side_effect=AssertionError("explicit backend"),
            ):
                view = build_run_speakers(
                    root, mode="required", config=config, legacy_builder=legacy
                )
            self.assertEqual(seen, ["segment-clustering"])
            self.assertEqual(view["status"], "partial")

    def test_speaker_legacy_ai_flag_uses_guarded_polish_without_implicit_summary(self):
        from workbench.enhancement import enhance_transcript_artifacts

        with (
            tempfile.TemporaryDirectory() as d,
            patch.dict(os.environ, {"NEXTECHO_RUNTIME_DIR": str(Path(d) / "runtime")}),
        ):
            root = Path(d) / "guarded"

            def asr(*args, output_dir, **kwargs):
                return asr_fixture(output_dir)

            def guarded(folder, **kwargs):
                self.assertFalse(kwargs["summarize"])
                self.assertEqual(kwargs["profile"], "fast")
                return enhance_transcript_artifacts(folder, ai_polish=False)

            with (
                patch(
                    "workbench.transcription._transcribe_media_sources", side_effect=asr
                ),
                patch(
                    "workbench.acoustic_dialogue.build_acoustic_dialogue",
                    side_effect=native_measurement_fixture,
                ),
                patch(
                    "workbench.postprocess.process_transcript_artifacts",
                    side_effect=guarded,
                ) as model,
                patch(
                    "workbench.cli.run_doctor",
                    return_value=SimpleNamespace(memory_gb=64),
                ),
                patch("workbench.cli.resolve_sources", return_value=[]),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                code = cli.main(
                    [
                        "transcribe",
                        "https://example.org/synthetic",
                        "--speaker-mode",
                        "required",
                        "--postprocess-profile",
                        "off",
                        "--ai-polish",
                        "--output-dir",
                        str(root),
                        "--json",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertEqual(model.call_count, 1)
            self.assertIn(
                "李继刚：", Path(verify_publication(root)["primary_file"]).read_text()
            )

    def test_legacy_ai_flag_without_speakers_also_uses_guarded_local_stage(self):
        with (
            tempfile.TemporaryDirectory() as d,
            patch.dict(os.environ, {"NEXTECHO_RUNTIME_DIR": str(Path(d) / "runtime")}),
        ):
            root = Path(d) / "no-speakers"

            def asr(*args, output_dir, **kwargs):
                return asr_fixture(output_dir)

            with (
                patch(
                    "workbench.transcription._transcribe_media_sources", side_effect=asr
                ),
                patch(
                    "workbench.postprocess.process_transcript_artifacts",
                    return_value={"status": "ready", "paths": {}},
                ) as guarded,
                patch(
                    "workbench.cli.run_doctor",
                    return_value=SimpleNamespace(memory_gb=64),
                ),
                patch("workbench.cli.resolve_sources", return_value=[]),
                patch(
                    "workbench.enhancement.maybe_ai_polish",
                    side_effect=AssertionError("legacy free-form command must not run"),
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                code = cli.main(
                    [
                        "transcribe",
                        "https://example.org/synthetic",
                        "--speaker-mode",
                        "off",
                        "--postprocess-profile",
                        "off",
                        "--ai-polish",
                        "--output-dir",
                        str(root),
                        "--json",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertEqual(guarded.call_count, 1)
            self.assertFalse(guarded.call_args.kwargs["summarize"])
            self.assertEqual(guarded.call_args.kwargs["profile"], "fast")


if __name__ == "__main__":
    unittest.main()
