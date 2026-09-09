from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import (
    JOBS,
    ENHANCEMENT_JOB,
    SPEAKER_TRANSCRIPT_JOB,
    TRANSCRIPTION_JOB,
    _new_job,
    _speaker_payload_v3,
    _write_raw_diarization_artifacts,
    app,
    reconcile_job_state,
    run_enhancement_job,
)


class AppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = app.test_client()
        JOBS.clear()

    def test_health_endpoint(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["ok"], True)
        self.assertEqual(payload["service"], "NextEcho")

    def test_raw_diarization_recovery_never_uses_display_name_as_raw_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            audio = root / "voice.wav"
            audio.write_bytes(b"voice")
            payload = {
                "backend": "test",
                "preprocess": {"voice_reference_audio": str(audio)},
                "speaker_turns": [
                    {"raw_speaker": "SPEAKER_00", "speaker": "孟岩", "start": 0.0, "end": 1.0}
                ],
            }
            _write_raw_diarization_artifacts(root, payload)
            raw = json.loads((root / "diarization.raw.json").read_text(encoding="utf-8"))
            self.assertEqual(raw["turns"][0]["raw_speaker_label"], "SPEAKER_00")
            self.assertNotIn("孟岩", (root / "diarization.raw.rttm").read_text(encoding="utf-8"))

    def test_raw_diarization_recovery_rejects_display_only_turns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(RuntimeError, "anonymous raw speaker label"):
                _write_raw_diarization_artifacts(
                    Path(tmp_dir),
                    {"speaker_turns": [{"speaker": "孟岩", "start": 0.0, "end": 1.0}]},
                )

    def test_speaker_payload_v3_keeps_canonical_identity_fields(self) -> None:
        result = _speaker_payload_v3(
            {
                "speaker_turns": [{"raw_speaker": "SPEAKER_00", "speaker": "孟岩", "start": 0.0, "end": 1.0}],
                "speakers": [
                    {
                        "speaker_id": "spk_00",
                        "raw_labels": ["SPEAKER_00"],
                        "display_name": "孟岩",
                        "confidence": 0.98,
                        "identity_status": "resolved",
                        "identity_evidence": [{"type": "self_introduction"}],
                    }
                ],
            }
        )
        speaker = result["speakers"][0]
        self.assertEqual(speaker["speaker_id"], "spk_00")
        self.assertEqual(speaker["confidence"], 0.98)
        self.assertEqual(speaker["identity_evidence"], [{"type": "self_introduction"}])

    def test_models_status_is_local_and_reports_profiles(self) -> None:
        with patch("app.ollama_status", return_value={"ready": True, "models": ["qwen3:4b"]}):
            response = self.client.get("/api/models/status")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertIn("Qwen/Qwen3-ASR-1.7B", payload["asr_models"].values())
        self.assertEqual(payload["llm_models"]["fast"], "qwen3:4b")

    def test_models_download_rejects_unknown_kind(self) -> None:
        response = self.client.post("/api/models/download", json={"kind": "remote-model"})
        self.assertEqual(response.status_code, 400)

    def test_reconcile_transcription_job_auto_completes_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dir = root / "run_demo"
            run_dir.mkdir(parents=True)
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "started_at": "2026-06-14T00:00:00Z",
                        "completed_at": "2026-06-14T00:01:00Z",
                        "elapsed_seconds": 60,
                        "results": [],
                        "error_count": 0,
                    }
                ),
                encoding="utf-8",
            )
            job = {
                "job_id": "job-1",
                "job_type": TRANSCRIPTION_JOB,
                "run_id": "run_demo",
                "status": "running",
                "elapsed_seconds": 1,
            }
            with patch("app.RUNS_ROOT", root):
                reconcile_job_state(job)
            self.assertEqual(job["status"], "completed")
            self.assertEqual(job["stage"], "complete")

    def test_reconcile_speaker_job_waits_for_speaker_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dir = root / "run_demo"
            run_dir.mkdir(parents=True)
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "started_at": "2026-06-14T00:00:00Z",
                        "completed_at": "2026-06-14T00:01:00Z",
                        "elapsed_seconds": 60,
                        "results": [],
                        "error_count": 0,
                    }
                ),
                encoding="utf-8",
            )
            job = {
                "job_id": "job-2",
                "job_type": SPEAKER_TRANSCRIPT_JOB,
                "run_id": "run_demo",
                "status": "running",
                "elapsed_seconds": 1,
            }
            with patch("app.RUNS_ROOT", root):
                reconcile_job_state(job)
            self.assertEqual(job["status"], "running")

            (run_dir / "transcript.speakers.json").write_text(
                json.dumps({"preview_text": "Speaker 1: hello"}),
                encoding="utf-8",
            )
            (run_dir / "transcript.speakers.txt").write_text("Speaker 1: hello", encoding="utf-8")
            (run_dir / "transcript.speakers.md").write_text("Speaker 1: hello", encoding="utf-8")
            with patch("app.RUNS_ROOT", root):
                reconcile_job_state(job)
            self.assertEqual(job["status"], "completed")
            self.assertEqual(job["message"], "访谈逐字稿已完成")

    def test_list_runs_prefers_real_episode_title(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dir = root / "run_demo"
            run_dir.mkdir(parents=True)
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "generated_at": "2026-06-14T00:00:00Z",
                        "model_name": "base",
                        "error_count": 0,
                        "results": [
                            {
                                "title": "真实节目标题",
                                "source_label": "001_demo",
                            }
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            with patch("app.RUNS_ROOT", root):
                response = self.client.get("/api/runs")
            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            self.assertEqual(payload["runs"][0]["display_title"], "真实节目标题")
            self.assertEqual(payload["runs"][0]["quality"], "更快")

    def test_historical_enhance_passes_effective_config_to_worker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_dir = root / "run_demo"
            item_dir = run_dir / "items" / "episode"
            item_dir.mkdir(parents=True)
            (item_dir / "transcript.json").write_text("{}", encoding="utf-8")
            (run_dir / "manifest.json").write_text(
                json.dumps({"results": [{"source_label": "episode", "output_dir": str(item_dir)}]}),
                encoding="utf-8",
            )
            job_id = "enhance-config"
            JOBS[job_id] = _new_job(job_id, ENHANCEMENT_JOB, run_id=run_dir.name)
            effective_config = {"postprocess": {"preferred_model_id": "configured:4b"}}
            seen: dict[str, object] = {}

            def fake_process(item_path: Path, **kwargs: object) -> dict:
                seen["item_dir"] = item_path
                seen["config"] = kwargs.get("config")
                return {"status": "ready", "paths": {}, "actual_models": ["configured:4b"]}

            with patch("app.RUNS_ROOT", root), patch("app.load_config", return_value=effective_config), patch(
                "app.process_transcript_artifacts", side_effect=fake_process
            ):
                run_enhancement_job(job_id, run_dir, "fast", "", "")

            self.assertIs(seen["config"], effective_config)
            self.assertEqual(seen["item_dir"], item_dir)
            self.assertEqual(JOBS[job_id]["status"], "ready")

    def test_model_status_uses_effective_config_selection(self) -> None:
        effective_config = {
            "asr": {"engine": "qwen3-asr", "accurate_model": "configured/asr", "fast_model": "configured/fast", "forced_aligner": "configured/aligner"},
            "diarization": {"backend": "sherpa-onnx", "fallback_order": ["sherpa-onnx", "resemblyzer"]},
            "postprocess": {
                "profile": "quality",
                "provider": "ollama",
                "preferred_model_id": "configured:4b",
                "fallback_model": "configured:4b",
                "quality_model": "configured:9b",
            },
        }
        with patch("app.load_config", return_value=effective_config), patch("app.qwen_accurate_model", return_value="configured/asr"), patch(
            "app.qwen_fast_model", return_value="configured/fast"
        ), patch("app.qwen_forced_aligner_model", return_value="configured/aligner"), patch(
            "app.qwen_mlx_available", return_value=True
        ), patch("app.qwen_memory_safe", return_value=True), patch("app.qwen_model_cached", return_value=True), patch(
            "app.qwen_forced_aligner_cached", return_value=True
        ), patch("app.qwen_model_provenance", return_value={"revision": "test-revision"}), patch(
            "app.detect_speaker_backend_status", return_value={"ready": True, "selected_backend": "sherpa-onnx"}
        ), patch(
            "app.ollama_status", return_value={"ready": True, "models": ["configured:4b", "configured:9b"]}
        ):
            response = self.client.get("/api/models/status")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["selected"]["asr"]["model"], "configured/asr")
        self.assertEqual(payload["selected"]["llm"]["model"], "configured:4b")
        self.assertEqual(payload["llm_models"]["quality"], "configured:9b")
        self.assertEqual(payload["diarization"]["requested_backend"], "sherpa-onnx")


if __name__ == "__main__":
    unittest.main()
