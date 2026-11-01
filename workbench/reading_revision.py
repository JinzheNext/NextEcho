"""Repackage a published run with current reading rules; no model execution.

Usage: python -m workbench.reading_revision /absolute/published/run
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .contracts import PipelineError
from .identity_revision import _copy_artifact, rebase_paths
from .io_utils import file_hash, write_json
from .publication import begin_pipeline, publish_pipeline, verify_publication
from .reading_view import VERSION


def publish_reading_revision(base: Path) -> dict:
    base = base.expanduser().resolve()
    original = verify_publication(base)
    if original is None:
        raise PipelineError("INVALID_INPUT", "阅读修订需要已校验发布的来源任务")
    if any(p.is_symlink() for p in base.rglob("*")):
        raise PipelineError("PROTOCOL_ERROR", "阅读修订不复制符号链接")
    started, revision = time.monotonic(), uuid4().hex
    target = base.parent / (datetime.now().strftime("run_%Y%m%d_reading_") + revision[:8])
    target.mkdir()
    try:
        shutil.copytree(base, target, copy_function=_copy_artifact, dirs_exist_ok=True)
        verify_publication(base, job_id=original["job_id"])
        for name, digest in original["artifact_hashes"].items():
            if file_hash(target / Path(name).relative_to(base)) != digest:
                raise PipelineError("CACHE_MISMATCH", "阅读修订副本校验失败", blocked=True)
        manifest = rebase_paths(json.loads((base / "manifest.json").read_text()), base, target)
        results = manifest["results"]
        aggregate_path = target / "transcript.speakers.json"
        if aggregate_path.is_file():
            aggregate = json.loads(aggregate_path.read_text())
            if aggregate.get("schema_name") == "nextecho.speaker-aggregate":
                from .speaker_service import relocated_speaker_view
                from .speaker_artifacts import write_v3_outputs
                write_v3_outputs(target, relocated_speaker_view(target, aggregate))
        record = {"schema_version": 1, "operation": "reading-revision", "reading_version": VERSION,
                  "base_run_dir": str(base), "base_job_id": original["job_id"],
                  "base_receipt_sha256": file_hash(base / "receipt.json"),
                  "asr_calls": 0, "speaker_calls": 0, "polish_calls": 0,
                  "raw_text_edited": False, "adopted_polish_edited": False}
        write_json(target / "reading.source.json", record)
        for item in results:
            item["reading_revision_path"] = str(target / "reading.source.json")
        manifest["reading_revision"] = record
        write_json(target / "manifest.json", manifest)
        begin_pipeline(target, job_id=revision)
        receipt = publish_pipeline(target, job_id=revision, results=results, plan=record,
            stages={"reading": "completed", "asr": "skipped", "speakers": "skipped", "polish": "skipped"},
            elapsed_seconds=time.monotonic() - started)
        verify_publication(target, job_id=revision)
        verify_publication(base, job_id=original["job_id"])
        return receipt
    except BaseException:
        shutil.rmtree(target)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    from .delivery import compact
    print(json.dumps(compact(publish_reading_revision(args.run_dir)), ensure_ascii=False))


if __name__ == "__main__":
    main()
