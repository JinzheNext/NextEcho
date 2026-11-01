"""Publish manual names as a new delivery, preserving the original execution."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from uuid import uuid4

from .contracts import PipelineError
from .io_utils import file_hash, write_json
from .publication import begin_pipeline, publish_pipeline, verify_publication
from .speaker_artifacts import (
    dialogue_to_v3,
    is_dialogue,
    mapped_dialogue,
    mapping_for_dialogue,
    write_identity_reading,
    write_v3_outputs,
)
from .speaker_scope import public_speaker_key, scope_from_files


def rebase_paths(value, source: Path, target: Path):
    if isinstance(value, dict):
        return {
            key: rebase_paths(child, source, target) for key, child in value.items()
        }
    if isinstance(value, list):
        return [rebase_paths(child, source, target) for child in value]
    if isinstance(value, str) and Path(value).is_absolute():
        try:
            relative = Path(value).resolve().relative_to(source)
        except (OSError, ValueError):
            return value
        return str(target / relative)
    return value


def _copy_artifact(source, target):
    source = Path(source)
    if source.is_symlink():
        raise PipelineError("PROTOCOL_ERROR", "身份修订不复制符号链接")
    # Media are immutable, hash checked inputs. Small metadata and text must
    # have separate inodes, including for legacy writers that overwrite files.
    if source.suffix.lower() in {
        ".wav",
        ".mp3",
        ".m4a",
        ".flac",
        ".mp4",
        ".ogg",
        ".opus",
        ".webm",
        ".mov",
        ".aac",
    }:
        try:
            os.link(source, target)
            return target
        except OSError:
            pass
    return shutil.copy2(source, target)


def publish_identity_revision(
    base: Path, updated: dict, mapping: dict
) -> tuple[Path, dict, dict]:
    started = time.monotonic()
    base = base.resolve()
    original = verify_publication(base)
    if original is None and (base / "pipeline.state.json").exists():
        raise PipelineError("INVALID_INPUT", "流水线尚未发布，不能提交人工身份修订")
    if any(path.is_symlink() for path in base.rglob("*")):
        raise PipelineError("PROTOCOL_ERROR", "身份修订来源包含符号链接")
    manifest_hash = file_hash(base / "manifest.json")
    revision_id = uuid4().hex
    prefix = base.name.encode()[:140].decode("utf-8", errors="ignore")
    target = base.parent / (prefix + "-identity-" + revision_id[:12])
    target.mkdir()  # Exclusive ownership before any failure cleanup.
    try:
        shutil.copytree(base, target, copy_function=_copy_artifact, dirs_exist_ok=True)
        # Recheck both the source and clone after copying, before changing any
        # derived file. A concurrent change cannot become a trusted revision.
        if file_hash(base / "manifest.json") != manifest_hash:
            raise PipelineError(
                "CACHE_MISMATCH", "复制时来源manifest发生变化", blocked=True
            )
        if original is not None:
            verify_publication(base, job_id=original["job_id"])
            for name, digest in original["artifact_hashes"].items():
                if file_hash(target / Path(name).relative_to(base)) != digest:
                    raise PipelineError(
                        "CACHE_MISMATCH", "身份修订副本校验失败", blocked=True
                    )
        view = rebase_paths(updated, base, target)
        nested = view.get("items") or []
        changes = {}
        identity_views = {}
        entries = nested or [
            {
                "payload": view,
                "paths": {"json": str(target / "transcript.speakers.json")},
            }
        ]
        for entry in entries:
            payload = entry["payload"]
            path = Path(entry["paths"]["json"])
            if not path.resolve().is_relative_to(target) or not path.is_file():
                raise PipelineError("PROTOCOL_ERROR", "身份修订条目路径越界或缺失")
            folder = path.parent
            ids = {person["speaker_id"] for person in payload["speakers"]}
            local = {
                identity: mapping[key]
                for identity in ids
                if (key := public_speaker_key(entry, identity))
                in mapping
            }
            canonical = folder / "transcript.dialogue.json"
            # The old writer put a nested dialogue at the legacy filename.
            # Migrate only the new revision; the source stays byte identical.
            old = json.loads(path.read_text(encoding="utf-8"))
            if not canonical.is_file() and is_dialogue(old):
                write_json(canonical, old)
            if canonical.is_file():
                data = json.loads(canonical.read_text(encoding="utf-8"))
                combined = {**mapping_for_dialogue(folder, data), **local}
                write_json(
                    folder / "speaker.identity-map.json",
                    {
                        "schema_version": 1,
                        "canonical_sha256": file_hash(canonical),
                        "mapping": combined,
                    },
                )
                payload = dialogue_to_v3(
                    mapped_dialogue(folder, data),
                    source_label=payload.get("source_label", ""),
                )
                payload["canonical_sha256"] = file_hash(canonical)
                payload = scope_from_files(payload, folder)
            paths = write_v3_outputs(folder, payload)
            if canonical.is_file():
                paths.update(
                    dialogue_json_path=str(canonical),
                    identity_mapping_path=str(folder / "speaker.identity-map.json"),
                )
            paths.update(write_identity_reading(folder, payload))
            changes[folder.resolve()] = paths
            identity_views[folder.resolve()] = payload
            if nested:
                entry["payload"] = payload
                entry["artifact_paths"] = paths
            else:
                view = {**payload, "paths": entry["paths"]}
        if nested:
            # Scoped IDs and corrected labels are already in the aggregate;
            # nested payloads retain recording-local IDs and bound evidence.
            view["speaker_map"] = {
                person["speaker_id"]: person["display_name"]
                for person in view["speakers"]
                if person["identity_status"] == "resolved"
            }
            write_v3_outputs(target, view)
            write_json(
                target / "speaker.identity-map.json",
                {
                    "schema_version": 1,
                    "scope": "run-local-identity",
                    "mapping": mapping,
                    "recordings": [
                        {
                            "source_label": entry["source_label"],
                            "canonical_sha256": entry["payload"].get(
                                "canonical_sha256", ""
                            ),
                            "identity_mapping_path": entry["artifact_paths"].get(
                                "identity_mapping_path", ""
                            ),
                        }
                        for entry in nested
                    ],
                },
            )
        manifest = rebase_paths(
            json.loads((base / "manifest.json").read_text()), base, target
        )
        results = manifest["results"]
        for item in results:
            folder = item.get("output_dir") or item.get("item_dir")
            folder = Path(folder) if folder else target / "items" / item["source_label"]
            if not folder.is_absolute():
                folder = target / folder
            if not folder.resolve().is_relative_to(target):
                raise PipelineError("PROTOCOL_ERROR", "身份修订manifest条目路径越界")
            if folder.resolve() in changes:
                item.update(changes[folder.resolve()])
                identity = identity_views[folder.resolve()]
                item["speaker_status"] = identity.get("status", "partial")
                item["speaker_identity"] = {
                    "schema_version": 3,
                    "status": identity.get("identity_status", "unknown"),
                    "speakers": identity["speakers"],
                    "mapping_path": changes[folder.resolve()].get(
                        "identity_mapping_path", ""
                    ),
                }
                if (folder / "metadata.json").exists():
                    write_json(folder / "metadata.json", item)
        # Single-item old runs may store artifacts at the root itself.
        if target in changes and len(results) == 1:
            results[0].update(changes[target])
        if results:
            results[0].update(
                identity_source_path=str(target / "identity.source.json"),
                aggregate_speaker_json_path=str(target / "transcript.speakers.json"),
                aggregate_speaker_text_path=str(target / "transcript.speakers.txt"),
                aggregate_speaker_markdown_path=str(target / "transcript.speakers.md"),
                aggregate_speaker_map_path=str(target / "speaker_map.json"),
            )
            if (target / "speaker.identity-map.json").is_file():
                results[0]["aggregate_identity_mapping_path"] = str(
                    target / "speaker.identity-map.json"
                )
        manifest["speaker_transcript"] = {
            "status": view.get("status", "partial"),
            "speakers": view["speakers"],
            "backend": view.get("backend", ""),
            "coverage": view.get("coverage", {}),
            "paths": view.get("paths", {}),
        }
        manifest["speaker_identity"] = {
            "schema_version": 3,
            "status": "resolved"
            if any(
                person["identity_status"] == "resolved" for person in view["speakers"]
            )
            else "unknown",
            "speakers": view["speakers"],
            "mapping_path": str(target / "speaker.identity-map.json")
            if (target / "speaker.identity-map.json").is_file()
            else "",
        }
        source_record = {
            "schema_version": 1,
            "operation": "manual-identity",
            "base_run_dir": str(base),
            "base_manifest_sha256": manifest_hash,
            "base_job_id": original.get("job_id", "") if original else "",
            "base_receipt_sha256": file_hash(base / "receipt.json") if original else "",
            "mapping": mapping,
            "models_called": 0,
            "text_edited": False,
        }
        manifest["identity_revision"] = source_record
        write_json(target / "manifest.json", manifest)
        write_json(target / "identity.source.json", source_record)
        begin_pipeline(target, job_id=revision_id)
        receipt = publish_pipeline(
            target,
            job_id=revision_id,
            results=results,
            plan=source_record,
            stages={"identity": "completed", "inference": "skipped"},
            elapsed_seconds=time.monotonic() - started,
        )
        verify_publication(target, job_id=revision_id)
        if file_hash(base / "manifest.json") != manifest_hash:
            raise PipelineError(
                "CACHE_MISMATCH", "发布期间来源manifest发生变化", blocked=True
            )
        if original is not None:
            verify_publication(base, job_id=original["job_id"])
        return target, view, receipt
    except BaseException:
        if target.is_dir():
            shutil.rmtree(target)
        raise
