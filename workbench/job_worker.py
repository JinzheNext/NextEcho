"""Detached job supervisor. Never requires the web server."""
from __future__ import annotations

import argparse
import json
import os
import signal
import threading
from pathlib import Path

from .contracts import PipelineError, PipelinePlan, PipelineRequest
from .delivery import assemble, write_receipt
from .io_utils import redact, write_json
from .job_store import JobStore
from .pipeline import execute_plan
from .publication import fail_pipeline


def run_job(store: JobStore, job_id: str):
    job = store.claim(job_id)
    if job is None:
        return
    os.environ["NEXTECHO_RUNTIME_DIR"] = str(store.root)
    os.environ["NEXTECHO_JOB_ID"] = job_id
    stopped = threading.Event()
    cancelling = threading.Event()
    def interrupted(*_):
        if cancelling.is_set():
            return
        cancelling.set()
        raise PipelineError("CANCELLED", "任务已取消")
    old_handler = signal.signal(signal.SIGTERM, interrupted)
    def heartbeat():
        while not stopped.wait(15):
            store.update(job_id, heartbeat_at=__import__("time").time())
    threading.Thread(target=heartbeat, daemon=True).start()
    run_dir = Path(job["run_dir"])
    try:
        raw = job["plan"]
        request = dict(raw["request"])
        request["sources"] = tuple(request["sources"])
        plan = PipelinePlan(PipelineRequest(**request), raw["asr_profile"], raw["polish_profile"], raw["effective"])
        if store.cancelled(job_id):
            interrupted()
        def progress(**fields):
            stage = fields.get("stage")
            if stage:
                fields["status"] = "queued" if stage == "queued" else "running"
            store.update(job_id, **fields)
        receipt = execute_plan(plan, run_dir, job_id=job_id, cancel=lambda: store.cancelled(job_id),
                   progress=progress, resume_from=job.get("resume_from", ""), resume_policy=job.get("resume_policy"))
    except Exception as exc:
        code = exc.code if isinstance(exc, PipelineError) else "DEPENDENCY_MISSING" if isinstance(exc, FileNotFoundError) else "PROTOCOL_ERROR"
        status = "cancelled" if code == "CANCELLED" else "blocked" if getattr(exc, "blocked", False) or isinstance(exc, FileNotFoundError) else "failed"
        fail_pipeline(run_dir, code=code, status=status)
        results = []
        if (run_dir / "manifest.json").is_file():
            try:
                results = json.loads((run_dir / "manifest.json").read_text()).get("results", [])
            except (OSError, ValueError, AttributeError):
                pass
        if any(r.get("text_path") and Path(r["text_path"]).is_file() for r in results):
            try:
                assemble(run_dir, results)
            except Exception as packaging_error:
                # A secondary rendering failure must not replace the actual
                # pipeline error with a later INTERRUPTED reconciliation.
                write_json(run_dir / "failure.assembly.json", {"original_error_code": code,
                    "assembly_error": redact(str(packaging_error))})
        receipt = write_receipt(run_dir, job_id=job_id, status=status, results=results, error_code=code, message=str(exc), plan=job["plan"], stage=getattr(exc, "stage", "prepare"))
    finally:
        stopped.set()
        signal.signal(signal.SIGTERM, old_handler)
    store.update(job_id, **{k: v for k, v in receipt.items() if k not in {"job_id", "plan"}})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    run_job(JobStore(args.root), args.job)


if __name__ == "__main__":
    main()
