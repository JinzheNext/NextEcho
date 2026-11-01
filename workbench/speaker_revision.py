"""Publish a speaker-only refinement without rerunning ASR or text polish."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import time
from pathlib import Path
from uuid import uuid4

from .contracts import PipelineError
from .io_utils import file_hash, value_hash, write_json
from .publication import begin_pipeline, publish_pipeline, verify_publication
from .speaker_service import build_run_speakers, resolve_item_dir

FROZEN_TEXT_INPUTS = (
    "transcript.json",
    "transcript.txt",
    "transcript.enhanced.json",
    "transcript.enhanced.txt",
    "postprocess.audit.json",
)
SPEAKER_DERIVED_FILES = (
    "transcript.speakers.json",
    "transcript.speakers.txt",
    "transcript.speakers.md",
    "speaker_map.json",
    "transcript.dialogue.json",
    "transcript.dialogue.txt",
    "diarization.raw.json",
    "diarization.raw.rttm",
    "speaker.identity-map.json",
)
SPEAKER_PATH_FIELDS = {
    "speaker_json_path", "speaker_text_path", "speaker_markdown_path", "speaker_map_path",
    "dialogue_json_path", "dialogue_text_path", "voice_linking_json_path",
    "diarization_raw_json_path", "diarization_raw_rttm_path", "speaker_identity_review_path",
    "speaker_refinement_path",
}
MEDIA_SUFFIXES = {".aac", ".flac", ".m4a", ".mp3", ".mp4", ".mov", ".ogg", ".opus", ".wav", ".webm"}


def _rebase(value, source: Path, target: Path):
    if isinstance(value, dict):
        return {key: _rebase(child, source, target) for key, child in value.items()}
    if isinstance(value, list):
        return [_rebase(child, source, target) for child in value]
    if isinstance(value, str) and Path(value).is_absolute():
        try:
            relative = Path(value).resolve().relative_to(source)
        except (OSError, ValueError):
            return value
        return str(target / relative)
    return value


def _copy_revision_artifact(source: str, target: str) -> str:
    source_path, target_path = Path(source), Path(target)
    if source_path.is_symlink():
        raise PipelineError("PROTOCOL_ERROR", "来源任务包含符号链接，拒绝复制")
    if source_path.suffix.lower() in MEDIA_SUFFIXES:
        try:
            os.link(source_path, target_path)
            return target
        except OSError:
            pass
    return shutil.copy2(source_path, target_path)


def _freeze_text_inputs(run_dir: Path, results: list[dict]) -> dict[str, str]:
    frozen: dict[str, str] = {}
    for item in results:
        folder = resolve_item_dir(run_dir, item)
        label = str(item.get("source_label") or folder.name)
        for name in FROZEN_TEXT_INPUTS:
            path = folder / name
            if path.is_file():
                frozen[f"{label}/{name}"] = file_hash(path)
        if item.get("status") in {"ready", "cached", "partial"} and item.get("char_count", 1):
            if not (folder / "transcript.json").is_file():
                raise PipelineError("INVALID_INPUT", f"来源缺少冻结的识别稿：{label}")
            if not any((folder / name).is_file() for name in ("transcript.txt", "transcript.json")):
                raise PipelineError("INVALID_INPUT", f"来源缺少可复用文字：{label}")
    return frozen


def _verified_copy_config(config, source: Path, target: Path):
    """Authorize only byte-verified stage-cache path relocation for this copy."""
    from .config import NextEchoConfig

    data = config.to_dict() if hasattr(config, "to_dict") else dict(config or {})
    runtime = data.get("runtime") or {}
    if not isinstance(runtime, dict):
        raise PipelineError("CONFIG_ERROR", "runtime配置必须为表")
    runtime = dict(runtime)
    runtime.update(
        resume_verified=True,
        resume_parent_dir=str(source.resolve()),
        resume_run_dir=str(target.resolve()),
    )
    data["runtime"] = runtime
    provenance = {
        key: getattr(config, key)
        for key in ("source_paths", "source_kinds", "repo_path", "user_path")
        if hasattr(config, key)
    }
    return NextEchoConfig(data, **provenance)


def _archive_prior_speaker_outputs(item_dir: Path, archive_root: Path) -> None:
    """Keep historical evidence while ensuring new immutable files start clean."""
    archived = False
    for name in SPEAKER_DERIVED_FILES:
        source = item_dir / name
        if source.is_file():
            archive_root.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(archive_root / name))
            archived = True
    stale_link = item_dir / "speaker-stages" / "voice-linking.json"
    if stale_link.is_file():
        archive_root.mkdir(parents=True, exist_ok=True)
        shutil.move(str(stale_link), str(archive_root / stale_link.name))
        archived = True
    old_review = item_dir / "speaker-review"
    if old_review.exists():
        archive_root.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_review), str(archive_root / "speaker-review"))
        archived = True
    if not archived and archive_root.exists():
        shutil.rmtree(archive_root)


def _write_review_pages(run_dir: Path, manifest: dict) -> None:
    from .identity_review import write_identity_review

    for item in manifest.get("results") or []:
        if item.get("status") not in {"ready", "cached", "partial"} or not item.get("char_count", 1):
            continue
        folder = resolve_item_dir(run_dir, item)
        dialogue = folder / "transcript.dialogue.json"
        if not dialogue.is_file():
            item["speaker_identity_review_status"] = "unavailable-no-dialogue-artifact"
            continue
        audio_value = item.get("audio_path") or item.get("normalized_audio_path") or ""
        audio = Path(audio_value).expanduser() if audio_value else folder / "audio.wav"
        if not audio.is_absolute():
            audio = (folder / audio).resolve()
        if not audio.is_file():
            for candidate in (folder / "转录音频.wav", folder / "audio.wav", folder / "audio.normalized.wav"):
                if candidate.is_file():
                    audio = candidate
                    break
        if not audio.is_file():
            item["speaker_identity_review_status"] = "unavailable-no-local-audio"
            continue
        review_dir = folder / "speaker-review"
        dialogue_data = json.loads(dialogue.read_text(encoding="utf-8"))
        origin_ms = dialogue_data.get("audio_origin_ms", item.get("audio_origin_ms", 0))
        try:
            result = write_identity_review(
                review_dir / "review.html", dialogue, audio, audio_origin_ms=origin_ms
            )
        except ValueError:
            # A local file alone is not trustworthy playback evidence; the
            # review writer also requires a separately bound expected digest.
            item["speaker_identity_review_status"] = "unavailable-audio-integrity"
            continue
        item["speaker_identity_review_path"] = str((review_dir / "review.html").resolve())
        item["speaker_identity_review_fingerprint"] = result["fingerprint"]
        item["speaker_identity_review_status"] = "human-review-pending"


def refine_published_speakers(
    base_run_dir: Path,
    *,
    backend: str = "auto",
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    config=None,
    progress_callback=None,
    idempotency_key: str = "",
) -> tuple[Path, dict, dict]:
    """Create a new immutable run with only the speaker stage recomputed.

    The speaker service consumes already stored transcript artifacts. This
    function never imports or calls ASR, polish, summary, or source download
    stages, and rejects publication if any frozen input changes.
    """
    started = time.monotonic()
    base = Path(base_run_dir).expanduser().resolve()
    original = verify_publication(base)
    if original is None:
        raise PipelineError("INVALID_INPUT", "仅支持已完整发布的来源任务")
    if config is None:
        from .config import load_config

        config = load_config()
    manifest_path = base / "manifest.json"
    source_manifest_hash = file_hash(manifest_path)
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_results = source_manifest.get("results")
    if not isinstance(source_results, list) or not source_results:
        raise PipelineError("INVALID_INPUT", "来源任务没有可复用的转录条目")
    frozen = _freeze_text_inputs(base, source_results)
    if not frozen:
        raise PipelineError("INVALID_INPUT", "没有找到可冻结的原稿或采用文字")
    if any(path.is_symlink() for path in base.rglob("*")):
        raise PipelineError("PROTOCOL_ERROR", "来源任务包含符号链接，拒绝复制")

    idempotency_key = str(idempotency_key or "").strip()
    if len(idempotency_key) > 200:
        raise PipelineError("INVALID_INPUT", "说话人精修幂等键超过长度限制")
    config_fingerprint = value_hash(config.to_dict() if hasattr(config, "to_dict") else (config or {}))
    request_fingerprint = value_hash({
        "base_manifest_sha256": source_manifest_hash,
        "backend": backend,
        "min_speakers": min_speakers,
        "max_speakers": max_speakers,
        "config_fingerprint": config_fingerprint,
    })
    revision_id = hashlib.sha256(idempotency_key.encode()).hexdigest()[:32] if idempotency_key else uuid4().hex
    prefix = base.name.encode()[:120].decode("utf-8", errors="ignore")
    target = base.parent / f"{prefix}-speakers-{revision_id[:12]}"
    if target.exists():
        existing = verify_publication(target, job_id=revision_id)
        if existing is None:
            raise PipelineError("INTERRUPTED", "相同说话人精修意图已有未完成attempt，请先检查其状态", blocked=True)
        existing_manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        existing_refinement = existing_manifest.get("speaker_refinement") or {}
        if (existing_refinement.get("request_fingerprint") != request_fingerprint
                or existing_refinement.get("idempotency_key") != idempotency_key):
            raise PipelineError("REQUEST_CONFLICT", "相同精修意图键对应不同来源或配置")
        return target, existing_manifest, existing
    target.mkdir()
    try:
        shutil.copytree(base, target, copy_function=_copy_revision_artifact, dirs_exist_ok=True)
        if file_hash(manifest_path) != source_manifest_hash:
            raise PipelineError("CACHE_MISMATCH", "复制时来源manifest发生变化", blocked=True)
        verify_publication(base, job_id=original["job_id"])
        for name, digest in original["artifact_hashes"].items():
            copied = target / Path(name).relative_to(base)
            if not copied.is_file() or file_hash(copied) != digest:
                raise PipelineError("CACHE_MISMATCH", "说话人精修副本校验失败", blocked=True)

        manifest = _rebase(source_manifest, base, target)
        manifest["results"] = _rebase(source_results, base, target)
        for item in manifest["results"]:
            folder = resolve_item_dir(target, item)
            metadata_path = folder / "metadata.json"
            if metadata_path.is_file():
                metadata = _rebase(json.loads(metadata_path.read_text(encoding="utf-8")), base, target)
                write_json(metadata_path, metadata)
            history = folder / "speaker-history" / f"base-{original['job_id']}"
            _archive_prior_speaker_outputs(folder, history)
            item["speaker_history_dir"] = str(history.resolve()) if history.is_dir() else ""
            for key in list(item):
                if key in SPEAKER_PATH_FIELDS or key.startswith("native_") and key.endswith("_path"):
                    item.pop(key, None)
        write_json(target / "manifest.json", manifest)
        begin_pipeline(target, job_id=revision_id)
        speaker_config = _verified_copy_config(config, base, target)

        if progress_callback:
            progress_callback({"stage": "speaker", "message": "复用冻结文字，仅执行说话人精修"})
        speaker_view = build_run_speakers(
            target,
            mode="required",
            config=speaker_config,
            backend=backend,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            progress_callback=progress_callback,
        )
        if not isinstance(speaker_view, dict) or speaker_view.get("status") not in {"ready", "partial"}:
            raise PipelineError("PROTOCOL_ERROR", "说话人阶段未返回可发布结果")

        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        _write_review_pages(target, manifest)
        actual_frozen = _freeze_text_inputs(target, manifest.get("results") or [])
        if actual_frozen != frozen:
            raise PipelineError("CACHE_MISMATCH", "说话人精修改变了冻结原稿或采用文字", blocked=True)
        for name, digest in frozen.items():
            source_label, filename = name.split("/", 1)
            source_item = next(
                (
                    row
                    for row in source_results
                    if str(row.get("source_label") or resolve_item_dir(base, row).name)
                    == source_label
                ),
                None,
            )
            if source_item is None:
                raise PipelineError("CACHE_MISMATCH", "无法核对冻结文字来源", blocked=True)
            source_path = resolve_item_dir(base, source_item) / filename
            if not source_path.is_file() or file_hash(source_path) != digest:
                raise PipelineError("CACHE_MISMATCH", "来源文字在精修期间发生变化", blocked=True)

        speaker_summary = manifest.get("speaker_transcript") or {}
        is_candidate = any(
            (entry.get("payload") or {}).get("release_status") == "candidate"
            for entry in speaker_summary.get("items", []) if isinstance(entry, dict)
        ) or speaker_summary.get("release_status") == "candidate"
        review_paths = [item.get("speaker_identity_review_path", "") for item in manifest["results"]]
        refinement = {
            "schema_version": 1,
            "operation": "speaker-only-refinement",
            "base_run_dir": str(base),
            "base_job_id": original["job_id"],
            "base_manifest_sha256": source_manifest_hash,
            "base_receipt_sha256": file_hash(base / "receipt.json"),
            "idempotency_key": idempotency_key,
            "request_fingerprint": request_fingerprint,
            "backend": backend,
            "stage_allowlist": ["diarization", "alignment", "identity-projection", "render", "review-page"],
            "stage_blocklist": ["source-resolution", "download", "asr", "polish", "summary"],
            "models_called": {"asr": 0, "polish": 0, "summary": 0},
            "frozen_text_artifacts": frozen,
            "quality_validation": "pending-author-gold" if is_candidate else "pending-author-review",
            "release_status": "candidate" if is_candidate else "unvalidated",
            "review_pages": [path for path in review_paths if path],
            "started_elapsed_seconds": round(max(0, time.monotonic() - started), 3),
        }
        manifest["speaker_refinement"] = refinement
        for item in manifest["results"]:
            item["speaker_refinement_path"] = str((target / "speaker-refinement.json").resolve())
        if is_candidate:
            for item in manifest["results"]:
                if item.get("speaker_status") == "ready":
                    item["speaker_status"] = "partial"
        write_json(target / "speaker-refinement.json", refinement)
        write_json(target / "manifest.json", manifest)
        if file_hash(base / "manifest.json") != source_manifest_hash:
            raise PipelineError("CACHE_MISMATCH", "发布期间来源manifest发生变化", blocked=True)
        verify_publication(base, job_id=original["job_id"])
        receipt = publish_pipeline(
            target,
            job_id=revision_id,
            results=manifest["results"],
            plan={"request": refinement},
            stages={"asr": "skipped", "speaker": "completed", "polish": "skipped"},
            elapsed_seconds=time.monotonic() - started,
        )
        verify_publication(target, job_id=revision_id)
        return target, json.loads((target / "manifest.json").read_text(encoding="utf-8")), receipt
    except BaseException:
        if target.is_dir():
            shutil.rmtree(target)
        raise
