"""Auditable exact/config-refresh plans; source selection and date stay frozen."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .contracts import PipelineError, PipelinePlan, PipelineRequest, plan_request
from .io_utils import file_hash, redact_structure, value_hash


def capture_resume_inputs(parent: dict) -> dict:
    root = Path(parent["run_dir"]).resolve()
    ledger = parent.get("artifact_hashes") or {}
    paths = {root / name for name in ("delivery.context.json", "resolved_sources.json")}
    paths.update((root / "items").glob("*/source.context.json"))
    paths.update(Path(p) for p in ledger if Path(p).name == "source.context.json" and Path(p).is_absolute())
    media = {}
    for checkpoint in (root / "items").glob("*/audio.checkpoint.json"):
        try:
            data = json.loads(checkpoint.read_text())
            name = data["media_name"]
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError("invalid media basename")
            candidate = checkpoint.parent / name
            if not candidate.is_file():
                raise ValueError("frozen media missing")
            paths.add(candidate)
            media[str(checkpoint.parent.relative_to(root))] = {"relative_path": str(candidate.relative_to(root)),
                "sha256": data["media_sha256"]}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise PipelineError("CACHE_MISMATCH", "已捕获的媒体身份无法核验，不能重新下载替换冻结内容", blocked=True) from exc
    files = {}
    for path in sorted(paths):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise PipelineError("CACHE_MISMATCH", "恢复来源快照含外部或符号链接文件", blocked=True)
        digest = file_hash(path) if path.is_file() else None
        expected = ledger.get(str(path.resolve()))
        if expected and digest != expected:
            raise PipelineError("CACHE_MISMATCH", "冻结日期或来源快照已改变；不能冒充exact恢复", blocked=True)
        files[str(path.relative_to(root))] = digest
    if any(files[row["relative_path"]] != row["sha256"] for row in media.values()):
        raise PipelineError("CACHE_MISMATCH", "已捕获的原媒体hash改变，不能刷新为另一版本", blocked=True)
    return {"schema_version": 1, "basis": "parent-current-bytes-with-available-ledger-checks", "files": files,
            "media": media,
            "selected_sources_captured": bool(files.get("resolved_sources.json"))}


def verify_resume_inputs(root: Path, snapshot: dict):
    current = capture_resume_inputs({"run_dir": str(root)})
    if current["files"] != snapshot.get("files") or current["media"] != snapshot.get("media", {}):
        raise PipelineError("CACHE_MISMATCH", "来源快照在恢复提交后改变，尚未复制或重新采集", blocked=True)


def _changes(before, after, prefix=()):
    if isinstance(before, dict) and isinstance(after, dict):
        result = []
        for key in sorted(before.keys() | after.keys()):
            if key not in before or key not in after:
                result.append({"path": list((*prefix, key)), "before_present": key in before,
                    "after_present": key in after, "before": before.get(key), "after": after.get(key)})
            else:
                result.extend(_changes(before[key], after[key], (*prefix, key)))
        return result
    return [] if before == after else [{"path": list(prefix), "before_present": True,
                                        "after_present": True, "before": before, "after": after}]


def _cache_rechecks(original, updated, differences):
    stages = set()
    order = ["asr", "alignment", "speaker", "polish", "summary", "reading"]
    asr_profiles = {original["asr_profile"], updated["asr_profile"]}
    polish_profiles = {original["polish_profile"], updated["polish_profile"]}
    for row in differences:
        path = row["path"]
        if (path[:1] == ["asr_profile"] or path[:2] in (["effective", "asr"], ["effective", "context"])
                or path[:3] in (["effective", "resolved", "asr_model"], ["effective", "resolved", "asr_engine"])
                or path[:2] == ["effective", "models"] and len(path) > 2 and path[2] in asr_profiles):
            stages.update(order)
        elif path[:2] in (["effective", "diarization"], ["effective", "speaker_identity"]):
            stages.update(order[1:])
        elif (path[:1] == ["polish_profile"] or path[:2] in (["effective", "postprocess"], ["effective", "providers"])
                or path[:2] == ["effective", "models"] and len(path) > 2 and path[2] in polish_profiles
                or path[:3] in (["effective", "resolved", "polish_model"], ["effective", "resolved", "polish_provider"])):
            stages.update(order[3:])
        elif path[:2] in (["effective", "hardware"], ["effective", "runtime"]):
            stages.update(order)  # resource/budget checks, not automatic discard
        else:
            stages.add("reading")
    return [stage for stage in order if stage in stages]


def resume_plan(parent: dict, *, mode="exact", config=None, repair_polish=False) -> tuple[dict, dict]:
    if mode not in {"exact", "refresh"}:
        raise PipelineError("INVALID_INPUT", "恢复模式必须为exact或refresh")
    if repair_polish and mode != "refresh":
        raise PipelineError("INVALID_INPUT", "按句修复润色须显式使用--mode refresh")
    original = copy.deepcopy(parent["plan"])
    try:
        if original.get("schema_version", 1) != 1:
            raise ValueError("unsupported plan")
        request = dict(original["request"])
        request["sources"] = tuple(request["sources"])
        parsed = PipelineRequest(**request)
        parsed.validate()
        if not isinstance(original["effective"], dict):
            raise ValueError("invalid effective plan")
        if mode == "refresh":
            if config is None:
                raise ValueError("refresh config missing")
            updated = plan_request(parsed, config).to_dict()
            if repair_polish:
                updated["effective"].setdefault("postprocess", {})["repair_candidates"] = True
        else:
            # Do not reload the user's now-edited config for an exact attempt.
            updated = PipelinePlan(parsed, original["asr_profile"], original["polish_profile"], original["effective"]).to_dict()
    except (KeyError, TypeError, ValueError) as exc:
        raise PipelineError("PROTOCOL_ERROR", "旧任务计划无法核验，不能恢复", blocked=True) from exc
    # Normalize JSON tuple/list representation without changing actual values.
    updated = json.loads(json.dumps(updated))
    original = json.loads(json.dumps(original))
    differences = _changes(original, updated)
    # Diff values are local diagnostics; apply redaction under their original
    # keys rather than hiding a credential under a generic 'before' key.
    for row in differences:
        key = row["path"][-1]
        for side in ("before", "after"):
            row[side] = redact_structure({key: row[side]})[key]
    audit = {"schema_version": 1, "artifact_kind": "resume-plan-audit", "mode": mode,
        "parent_job_id": parent["job_id"], "parent_plan_sha256": value_hash(original),
        "effective_plan_sha256": value_hash(updated), "configuration_changed": original != updated,
        "differences": differences, "source_policy": "frozen", "date_policy": "inherit-original",
        "configuration_affected_stage_rechecks": _cache_rechecks(original, updated, differences),
        "cache_policy": "revalidate-stage-fingerprints-and-output-hashes",
        "cache_note": "配置刷新不直接授予缓存有效性；各阶段按实际输入/模型/代码/guard重新核验，旧accepted不能只因exact而免检。"}
    if parent.get("run_dir"):
        audit["source_snapshot"] = capture_resume_inputs(parent)
    return updated, audit
