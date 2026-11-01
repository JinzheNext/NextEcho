"""Actual publication and factory integration; inference is replaced at its boundary."""

from __future__ import annotations
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from app import (
    JOBS,
    _new_job,
    app,
    reconcile_job_state,
    serialize_run,
    update_job,
    TRANSCRIPTION_JOB,
)
from workbench.contracts import PipelineError
from workbench.io_utils import file_hash, write_json
from workbench.job_store import JobStore
from workbench.publication import publish_pipeline, verify_publication
from workbench.sources import SourceDescriptor
from workbench.transcription import _build_metadata, transcribe_media_sources


def factory_item(root, name="one", status="ready"):
    item = root / "items" / name
    item.mkdir(parents=True)
    if status == "ready":
        for filename in [
            "transcript.txt",
            "transcript.json",
            "audio.wav",
            "source.mp3",
        ]:
            (item / filename).write_text(
                "原始全文" if filename.endswith(".txt") else "{}"
            )
    descriptor = SourceDescriptor(
        "direct_media",
        "direct",
        "https://example.org/" + name,
        "https://example.org/" + name,
        "https://example.org/" + name + ".mp3",
        name,
        "",
        1,
        "",
        name,
        "direct",
    )
    result = _build_metadata(
        descriptor=descriptor,
        source_label=name,
        status=status,
        transcript_text="原始全文" if status == "ready" else "",
        item_dir=item,
        media_path=item / "source.mp3",
        audio_path=item / "audio.wav",
        created_at="2026-10-20T00:00:00Z",
        error="download failed" if status == "failed" else "",
        asr={"actual_engine": "whisper-cpp", "timing_precision": "segment"},
    )
    write_json(item / "metadata.json", result)
    return result


class PublicationTests(unittest.TestCase):
    def setUp(self):
        JOBS.clear()

    def test_real_metadata_factory_mixed_result_commits_partial(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            good, bad = factory_item(root), factory_item(root, "failed", "failed")
            self.assertFalse(Path(bad["media_path"]).exists())
            receipt = publish_pipeline(
                root,
                job_id="mixed",
                results=[good, bad],
                stages={"asr": "completed", "polish": "off"},
            )
            self.assertEqual(receipt["status"], "partial")
            self.assertIn("原始全文", Path(receipt["primary_file"]).read_text())
            self.assertEqual(verify_publication(root, job_id="mixed"), receipt)
            self.assertNotIn(bad["media_path"], receipt["artifact_hashes"])
            self.assertEqual(
                json.loads((root / "manifest.json").read_text())["error_count"], 1
            )
            self.assertEqual(
                json.loads((root / "errors.json").read_text())["items"][0][
                    "source_label"
                ],
                "failed",
            )

    def test_active_web_stages_and_item_failure_cannot_complete_from_asr_manifest(self):
        with tempfile.TemporaryDirectory() as d, patch("app.RUNS_ROOT", Path(d)):
            run = Path(d) / "run"
            results = [factory_item(run)]
            write_json(
                run / "manifest.json",
                {"completed_at": "2026-10-20T00:00:00Z", "results": results},
            )
            job = _new_job("web", TRANSCRIPTION_JOB, run_id="run")
            JOBS["web"] = job
            client = app.test_client()
            for stage in [
                "asr",
                "diarization",
                "polish",
                "summarize",
                "item_failed",
                "complete",
            ]:
                update_job("web", stage=stage, progress=100)
                response = client.get("/api/jobs/web").get_json()
                self.assertNotIn(
                    response["status"], {"ready", "partial", "completed"}, stage
                )
                self.assertLess(response["progress"], 100, stage)
                self.assertIsNone(response["result"])
            publish_pipeline(
                run,
                job_id="web",
                results=results,
                stages={
                    "asr": "completed",
                    "speaker": "completed",
                    "polish": "completed",
                },
            )
            response = client.get("/api/jobs/web").get_json()
            self.assertEqual(response["status"], "ready")
            self.assertEqual(response["progress"], 100)

    def test_queries_do_not_rewrite_committed_bytes_and_tamper_is_a_projection(self):
        with tempfile.TemporaryDirectory() as d, patch("app.RUNS_ROOT", Path(d)):
            run = Path(d) / "run"
            results = [factory_item(run)]
            receipt = publish_pipeline(run, job_id="web", results=results)
            JOBS["web"] = _new_job("web", TRANSCRIPTION_JOB, run_id="run")
            client = app.test_client()
            self.assertEqual(client.get("/api/jobs/web").get_json()["status"], "ready")
            frozen = {
                p: file_hash(p)
                for p in [
                    run / "manifest.json",
                    run / "receipt.json",
                    run / "pipeline.commit.json",
                ]
            }
            serialize_run(run)
            self.assertEqual(client.get("/api/jobs/web").get_json()["status"], "ready")
            self.assertEqual({p: file_hash(p) for p in frozen}, frozen)
            primary = Path(receipt["primary_file"])
            original = primary.read_bytes()
            primary.write_text("tampered")
            blocked = client.get("/api/jobs/web").get_json()
            self.assertEqual(blocked["status"], "blocked")
            self.assertEqual(blocked["execution_status"], "ready")
            self.assertEqual(JOBS["web"]["status"], "ready")
            self.assertEqual(blocked["primary_file"], "")
            primary.write_bytes(original)
            self.assertEqual(client.get("/api/jobs/web").get_json()["status"], "ready")

    def test_old_generation_cannot_complete_new_job_in_same_run(self):
        with tempfile.TemporaryDirectory() as d, patch("app.RUNS_ROOT", Path(d)):
            run = Path(d) / "run"
            publish_pipeline(run, job_id="old", results=[factory_item(run)])
            job = _new_job("new", TRANSCRIPTION_JOB, run_id="run")
            self.assertIsNone(verify_publication(run, job_id="new"))
            self.assertEqual(reconcile_job_state(job)["status"], "queued")

    def test_native_job_requires_final_marker_and_preserves_history_if_marker_removed(
        self,
    ):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            store = JobStore(root / "registry")
            plan = {"request": {}}
            job, _ = store.submit(plan, output_dir=root / "run")
            run = Path(job["run_dir"])
            receipt = publish_pipeline(
                run, job_id=job["job_id"], results=[factory_item(run)], plan=plan
            )
            saved = store.update(
                job["job_id"],
                **{k: v for k, v in receipt.items() if k not in {"job_id", "plan"}},
            )
            self.assertEqual(store.get(job["job_id"])["status"], "ready")
            (run / "pipeline.commit.json").unlink()
            self.assertEqual(store.get(job["job_id"])["status"], "blocked")
            self.assertEqual(store.get(job["job_id"], reconcile=False), saved)

    def test_marker_is_last_and_missing_successful_artifact_prevents_publication(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            result = factory_item(root)
            actual_write = __import__(
                "workbench.publication", fromlist=["write_json"]
            ).write_json

            def interrupt(path, data):
                if path.name == "pipeline.commit.json":
                    raise OSError("interrupted before commit")
                return actual_write(path, data)

            with patch("workbench.publication.write_json", side_effect=interrupt):
                with self.assertRaises(OSError):
                    publish_pipeline(root, job_id="cut", results=[result])
            self.assertIsNone(verify_publication(root))
            Path(result["text_path"]).unlink()
            with self.assertRaises(PipelineError):
                publish_pipeline(root, job_id="missing", results=[result])
            self.assertIsNone(verify_publication(root))

    def test_legacy_adapter_waits_for_speaker_and_polish_before_publishing(self):
        with (
            tempfile.TemporaryDirectory() as d,
            patch.dict(os.environ, {"NEXTECHO_RUNTIME_DIR": str(Path(d) / "runtime")}),
        ):
            root = Path(d) / "run"
            item = factory_item(root)
            manifest = root / "manifest.json"
            write_json(manifest, {"results": [item]})
            payload = {"output_dir": root, "manifest_path": manifest, "results": [item]}
            order = []

            def speaker(_):
                self.assertIsNone(verify_publication(root))
                order.append("speaker")
                return {"status": "ready"}

            def polish(*args, **kwargs):
                self.assertIsNone(verify_publication(root))
                order.append("polish")
                return {"status": "ready", "paths": {}}

            with (
                patch(
                    "workbench.transcription._transcribe_media_sources",
                    return_value=payload,
                ),
                patch(
                    "workbench.postprocess.process_transcript_artifacts",
                    side_effect=polish,
                ),
            ):
                result = transcribe_media_sources(
                    [],
                    output_dir=root,
                    speaker_stage=speaker,
                    postprocess_profile="quality",
                )
            self.assertEqual(order, ["speaker", "polish"])
            self.assertEqual(verify_publication(root)["status"], "ready")
            self.assertEqual(result["receipt"]["publication_schema"], 1)

    def test_history_and_detail_use_stage_journal_and_do_not_rewrite_asr(self):
        from workbench.publication import (
            begin_pipeline,
            mark_pipeline_stage,
            fail_pipeline,
        )

        with tempfile.TemporaryDirectory() as d, patch("app.RUNS_ROOT", Path(d)):
            run = Path(d) / "run_stages"
            results = [factory_item(run)]
            write_json(
                run / "manifest.json", {"results": results, "completed_at": "asr-only"}
            )
            before = (run / "manifest.json").read_bytes()
            begin_pipeline(run, job_id="view")
            client = app.test_client()
            for phase in ["asr", "speaker", "polish", "package"]:
                mark_pipeline_stage(run, phase)
                self.assertNotIn(
                    client.get("/api/runs/run_stages").get_json()["manifest"]["status"],
                    {"ready", "partial"},
                )
                self.assertNotIn(
                    client.get("/api/runs").get_json()["runs"][0]["status"],
                    {"ready", "partial"},
                )
            self.assertEqual((run / "manifest.json").read_bytes(), before)
            fail_pipeline(run, code="STAGE_TIMEOUT")
            self.assertEqual(
                client.get("/api/runs/run_stages").get_json()["manifest"]["status"],
                "failed",
            )

    def test_native_pipeline_has_no_committed_generation_during_requested_stages(self):
        from workbench.pipeline import execute_plan
        from workbench.contracts import PipelineRequest, plan_request
        from workbench.config import NextEchoConfig
        from workbench.model_profiles import PROFILES

        with (
            tempfile.TemporaryDirectory() as d,
            patch.dict(os.environ, {"NEXTECHO_RUNTIME_DIR": str(Path(d) / "runtime")}),
        ):
            root = Path(d) / "run"
            root.mkdir()
            source = Path(d) / "input.wav"
            source.write_bytes(b"synthetic source")
            config = NextEchoConfig(
                {
                    "asr": {"engine": "sensevoice"},
                    "providers": {"ollama": {"ownership": "external"}},
                }
            )
            plan = plan_request(
                PipelineRequest(
                    (str(source),), postprocess="polish", speaker_mode="required"
                ),
                config,
            )
            model = PROFILES[plan.polish_profile].model
            order = []

            def asr(*args, output_dir, **kwargs):
                self.assertTrue(kwargs["defer_delivery"])
                self.assertIsNone(verify_publication(root))
                order.append("asr")
                results = [factory_item(output_dir)]
                write_json(root / "manifest.json", {"results": results})
                return {"results": results}

            def speaker(*args, **kwargs):
                self.assertIsNone(verify_publication(root))
                order.append("speaker")
                stage = Path(args[0])
                # The inference boundary must emit the actual artifact
                # protocol; a status-only mock cannot prove publication.
                from workbench.speaker_artifacts import write_v3_outputs
                payload = {"schema_version":3,"status":"ready","speakers":[],
                    "speaker_turns":[],"speaker_map":{}}
                write_v3_outputs(stage,payload)
                write_json(stage/'diarization.raw.json',{'annotation_kind':'acoustic-diarization','turns':[]})
                (stage/'diarization.raw.rttm').write_text('')
                return payload

            def polish(item_dir, **kwargs):
                self.assertIsNone(verify_publication(root))
                order.append("polish")
                text = item_dir / "transcript.enhanced.txt"
                text.write_text("原始全文。")
                return {"status": "ready", "paths": {"enhanced_text_path": str(text)}}

            with (
                patch("workbench.pipeline.transcribe_media_sources", side_effect=asr),
                patch(
                    "workbench.bounded_speaker.bounded_build_speaker_transcript",
                    side_effect=speaker,
                ),
                patch(
                    "workbench.postprocess.process_transcript_artifacts",
                    side_effect=polish,
                ),
                patch(
                    "workbench.pipeline.request_json",
                    return_value={"models": [{"name": model, "size": 100}]},
                ),
                patch(
                    "workbench.pipeline.sample_resources",
                    return_value={
                        "available_bytes": 10**12,
                        "pressure": "normal",
                        "swap_used_bytes": 0,
                    },
                ),
            ):
                receipt = execute_plan(plan, root, job_id="native")
            self.assertEqual(order, ["asr", "speaker", "polish"])
            self.assertEqual(
                verify_publication(root, job_id="native"),
                json.loads(json.dumps(receipt)),
            )
            self.assertEqual(receipt["status"], "ready")
            self.assertIn("原始全文。", Path(receipt["primary_file"]).read_text())

    def test_legacy_failure_records_terminal_stage_while_owner_remains_alive(self):
        from workbench.publication import publication_view

        with (
            tempfile.TemporaryDirectory() as d,
            patch.dict(os.environ, {"NEXTECHO_RUNTIME_DIR": str(Path(d) / "runtime")}),
        ):
            root = Path(d) / "run"
            with patch(
                "workbench.transcription._transcribe_media_sources",
                side_effect=PipelineError(
                    "RESOURCE_BLOCKED", "synthetic resource failure", blocked=True
                ),
            ):
                with self.assertRaises(PipelineError):
                    transcribe_media_sources([], output_dir=root)
            self.assertIsNone(verify_publication(root))
            self.assertEqual(publication_view(root)["status"], "blocked")
            self.assertEqual(publication_view(root)["error_code"], "RESOURCE_BLOCKED")

    def test_polish_cancellation_is_not_published_as_partial_success(self):
        from workbench.publication import publication_view

        with (
            tempfile.TemporaryDirectory() as d,
            patch.dict(os.environ, {"NEXTECHO_RUNTIME_DIR": str(Path(d) / "runtime")}),
        ):
            root = Path(d) / "run"
            item = factory_item(root)
            write_json(root / "manifest.json", {"results": [item]})
            payload = {
                "output_dir": root,
                "manifest_path": root / "manifest.json",
                "results": [item],
            }
            with (
                patch(
                    "workbench.transcription._transcribe_media_sources",
                    return_value=payload,
                ),
                patch(
                    "workbench.postprocess.process_transcript_artifacts",
                    side_effect=PipelineError("CANCELLED", "synthetic cancellation"),
                ),
            ):
                with self.assertRaises(PipelineError) as error:
                    transcribe_media_sources(
                        [], output_dir=root, postprocess_profile="quality"
                    )
            self.assertEqual(error.exception.code, "CANCELLED")
            self.assertIsNone(verify_publication(root))
            self.assertEqual(publication_view(root)["status"], "cancelled")

    def test_live_web_owner_is_not_failed_by_old_heartbeat(self):
        with tempfile.TemporaryDirectory() as d, patch("app.RUNS_ROOT", Path(d)):
            job = _new_job("alive", TRANSCRIPTION_JOB, run_id="run")
            job.update(status="polishing", heartbeat_at=time.time() - 10000)
            self.assertEqual(reconcile_job_state(job)["status"], "polishing")


if __name__ == "__main__":
    unittest.main()
