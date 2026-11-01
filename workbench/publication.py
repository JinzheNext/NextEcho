"""Publish a complete pipeline generation last; queries only verify its bytes."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from .contracts import PipelineError
from .delivery import assemble, write_receipt
from .io_utils import file_hash, value_hash, write_json


def outcome(results: list[dict]) -> tuple[str, int, int]:
    failures = sum(
        r.get("status") in {"failed", "blocked", "cancelled"} for r in results
    )
    partials = sum(
        r.get("status") == "partial"
        or r.get("identity_render_status") == "partial"
        or r.get("speaker_status") == "partial"
        or (r.get("enhancement") or {}).get("status") == "partial"
        or bool((r.get("asr") or {}).get("fallbacks"))
        or bool((r.get("asr") or {}).get("review_required"))
        for r in results
    )
    status = (
        "failed"
        if not results or failures == len(results)
        else "partial"
        if failures or partials
        else "ready"
    )
    return status, failures, partials


def invalidate_publication(run_dir: Path) -> None:
    """A new attempt cannot inherit the previous generation's success marker."""
    (run_dir / "pipeline.commit.json").unlink(missing_ok=True)


def begin_pipeline(run_dir: Path, *, job_id: str) -> None:
    from .resources import process_birth

    invalidate_publication(run_dir)
    write_json(
        run_dir / "pipeline.state.json",
        {
            "schema_version": 1,
            "job_id": job_id,
            "status": "running",
            "stage": "prepare",
            "owner_pid": os.getpid(),
            "owner_birth": process_birth(os.getpid()) if os.name == "posix" else "",
        },
    )


def fail_pipeline(run_dir: Path, *, code: str, status: str = "failed") -> None:
    path = run_dir / "pipeline.state.json"
    if path.is_file():
        state = json.loads(path.read_text())
        state.update(status=status, stage="failed", error_code=code)
        write_json(path, state)


def mark_pipeline_stage(run_dir: Path, stage: str) -> None:
    path = run_dir / "pipeline.state.json"
    if path.is_file():
        state = json.loads(path.read_text())
        state.update(
            status="packaging" if stage == "package" else "running", stage=stage
        )
        write_json(path, state)


def publication_view(run_dir: Path) -> dict | None:
    state_path = run_dir / "pipeline.state.json"
    if not state_path.is_file() and not (run_dir / "pipeline.commit.json").is_file():
        return None  # Historical artifact, with no new execution claim.
    try:
        state = json.loads(state_path.read_text()) if state_path.is_file() else {}
        receipt = verify_publication(run_dir, job_id=state.get("job_id"))
        if receipt is not None:
            return {
                "status": receipt["status"],
                "primary_file": receipt.get("primary_file", ""),
                "stage": "complete",
            }
        if state.get("status") in {"failed", "blocked", "cancelled"}:
            return {
                "status": state["status"],
                "stage": "failed",
                "error_code": state.get("error_code", ""),
            }
        if state.get("owner_birth"):
            from .resources import process_alive

            if not process_alive(state["owner_pid"], state["owner_birth"]):
                return {
                    "status": "failed",
                    "stage": "failed",
                    "error_code": "INTERRUPTED",
                }
        return {
            "status": "packaging" if state.get("stage") == "package" else "running",
            "stage": state.get("stage", "prepare"),
        }
    except (OSError, ValueError, TypeError, AttributeError, PipelineError):
        return {"status": "blocked", "error_code": "CACHE_MISMATCH", "stage": "package"}


def publish_pipeline(
    run_dir: Path,
    *,
    job_id: str,
    results: list[dict],
    plan: dict | None = None,
    stages: dict | None = None,
    elapsed_seconds: float | None = None,
) -> dict:
    if not isinstance(results, list) or any(
        not isinstance(r, dict)
        or r.get("status")
        not in {"ready", "cached", "partial", "failed", "blocked", "cancelled"}
        for r in results
    ):
        raise PipelineError(
            "PROTOCOL_ERROR", "来源条目尚未到可交付阶段", stage="package"
        )
    status, failures, partials = outcome(results)
    stage_states = stages or {"asr": "completed"}
    if any(s not in {"completed", "off", "skipped"} for s in stage_states.values()):
        raise PipelineError(
            "PROTOCOL_ERROR", "请求的流水线阶段尚未结束", stage="package"
        )
    invalidate_publication(run_dir)
    mark_pipeline_stage(run_dir, "package")
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text()) if path.is_file() else {}
    manifest.setdefault("asr_completed_at", manifest.get("completed_at", ""))
    manifest.setdefault("asr_elapsed_seconds", manifest.get("elapsed_seconds", 0))
    stamp = datetime.now(UTC).isoformat()
    manifest.update(
        results=results,
        error_count=failures,
        partial_count=partials,
        status=status,
        completed_at=stamp,
    )
    manifest["publication"] = {"status": status, "stage": "complete"}
    manifest["pipeline"] = {
        "schema_version": 1,
        "status": status,
        "job_id": job_id,
        "stages": stage_states,
        "plan_fingerprint": value_hash(plan),
        "completed_at": stamp,
        "commit_path": str((run_dir / "pipeline.commit.json").resolve()),
    }
    if elapsed_seconds is not None:
        manifest["elapsed_seconds"] = round(max(0, elapsed_seconds))
    write_json(path, manifest)
    if failures < len(results):
        primary = assemble(run_dir, results)
        manifest["pipeline"]["primary_file"] = str(primary.resolve())
        write_json(path, manifest)
    receipt = write_receipt(
        run_dir,
        job_id=job_id,
        status=status,
        results=results,
        plan=plan,
        error_code="PARTIAL_RESULT"
        if status == "partial"
        else "ASR_FAILED"
        if status == "failed"
        else "",
        message="全文已保留；润色采用或说话人仍有待审核项，具体原因见回执completion" if status == "partial" else "",
    )
    receipt["publication_schema"] = 1
    write_json(run_dir / "receipt.json", receipt)
    write_json(
        run_dir / "pipeline.commit.json",
        {
            "schema_version": 1,
            "job_id": job_id,
            "status": status,
            "plan_fingerprint": value_hash(plan),
            "stages": stage_states,
            "receipt_sha256": file_hash(run_dir / "receipt.json"),
        },
    )
    return receipt


def verify_publication(run_dir: Path, *, job_id: str | None = None) -> dict | None:
    """None means unpublished; malformed or changed delivery is a hard mismatch."""
    marker = run_dir / "pipeline.commit.json"
    if not marker.is_file():
        return None
    try:
        commit = json.loads(marker.read_text())
        receipt_path = run_dir / "receipt.json"
        if (
            commit["schema_version"] == 1
            and job_id is not None
            and commit["job_id"] != job_id
        ):
            return None  # A previous generation cannot complete this attempt.
        if commit["schema_version"] != 1 or commit["receipt_sha256"] != file_hash(
            receipt_path
        ):
            raise ValueError("invalid publication identity")
        receipt = json.loads(receipt_path.read_text())
        if any(
            receipt.get(k) != commit.get(k)
            for k in ("job_id", "status", "plan_fingerprint")
        ):
            raise ValueError("receipt does not match commit")
        inventory = receipt["artifact_hashes"]
        if not isinstance(inventory, dict) or (
            receipt["status"] in {"ready", "partial"} and not inventory
        ):
            raise ValueError("missing artifact inventory")
        for name, digest in inventory.items():
            candidate = Path(name)
            if (
                not candidate.is_absolute()
                or not candidate.resolve().is_relative_to(run_dir.resolve())
                or file_hash(candidate) != digest
            ):
                raise ValueError("changed artifact")
        primary = receipt.get("primary_file")
        if receipt["status"] in {"ready", "partial"} and (
            not primary or primary not in inventory
        ):
            raise ValueError("missing primary")
        return receipt
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise PipelineError(
            "CACHE_MISMATCH",
            "流水线交付尚未通过完整性校验，请恢复为新attempt",
            blocked=True,
            stage="package",
        ) from exc
