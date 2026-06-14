from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import (
    JOBS,
    SPEAKER_TRANSCRIPT_JOB,
    TRANSCRIPTION_JOB,
    app,
    reconcile_job_state,
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


if __name__ == "__main__":
    unittest.main()
