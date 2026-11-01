"""Shared local pipeline. No Flask imports and no model stdout."""
from __future__ import annotations

import json
import time
import shutil
from pathlib import Path

from .config import NextEchoConfig
from .contracts import PipelineError, PipelinePlan
from .publication import publish_pipeline, begin_pipeline, mark_pipeline_stage
from .io_utils import file_hash, value_hash, write_json
from .llm_providers import request_json
from .model_profiles import PROFILES
from .resources import ExecutionLease, model_policy, sample_resources
from .sources import SourceDescriptor, resolve_media_inputs
from .transcription import transcribe_media_sources
from .reading_artifacts import freeze_delivery_context


def run_post_asr_stages(payload, *, speaker=None, polish=None, speaker_mode="auto", native_speaker_alignment=False):
    """Shared Web/CLI/Skill order: measured ASR -> speaker -> text processing."""
    speech = [r for r in payload.get("results", []) if r.get("status") in {"ready", "cached"} and r.get("char_count", 1)]
    coarse = [r for r in speech if (r.get("asr") or {}).get("actual_engine") in {"fun-asr-nano", "sensevoice"}
              and (r.get("asr") or {}).get("timing_precision") != "precise"]
    if speaker and speaker_mode != "off" and coarse and not native_speaker_alignment:
        if speaker_mode == "required":
            raise PipelineError("NEEDS_ALIGNER", "逐词说话人归属需要已验收时间轴；原始文字已保留", blocked=True, stage="speaker")
        speaker_result = {"status": "partial", "error_code": "NEEDS_ALIGNER", "message": "已跳过未验收时间轴上的说话人归属"}
        for row in coarse:
            row["speaker_status"] = "partial"
    else:
        speaker_result = speaker(payload) if speaker and speaker_mode != "off" and speech else None
    if polish:
        polish()
    return speaker_result


def execute_plan(plan: PipelinePlan, run_dir: Path, *, job_id: str, progress=lambda **kw: None,
                 cancel=lambda: False, resume_from: str = "", resume_policy: dict | None = None) -> dict:
    started = time.monotonic()
    begin_pipeline(run_dir, job_id=job_id)
    config = NextEchoConfig(plan.effective)
    freeze_delivery_context(run_dir, parent_dir=Path(resume_from) if resume_from else None)
    req = plan.request
    selected_config = config.to_dict()
    selected_config.setdefault("asr", {}).update(allow_fallback=req.allow_fallback, strict_selection=True)
    config = NextEchoConfig(selected_config)
    if resume_from:
        # Copy immutable historical input; never link mutable transcript files.
        old = Path(resume_from)
        from .resume import capture_resume_inputs, verify_resume_inputs
        snapshot = (resume_policy or {}).get("source_snapshot") or capture_resume_inputs({"run_dir": str(old)})
        verify_resume_inputs(old, snapshot)
        write_json(run_dir / "resume.inputs.json", snapshot)
        for name in ("items",):
            if (old / name).is_dir():
                shutil.copytree(old / name, run_dir / name, dirs_exist_ok=True)
        def relocate(value, key=""):
            if isinstance(value, dict):
                return {k: v if k in {"asr", "source_context", "provenance", "model_selection", "candidate_origin_path"} else relocate(v, k)
                        for k, v in value.items()}
            if isinstance(value, list):
                return [relocate(v, key) for v in value]
            path_field = key in {"path", "output_dir", "run_dir"} or key.endswith(("_path", "_file"))
            if path_field and isinstance(value, str) and value.startswith(str(old.resolve()) + "/"):
                return str(run_dir.resolve()) + value[len(str(old.resolve())):]
            if path_field and isinstance(value, str) and "/items/" in value:
                relative = "items/" + value.split("/items/", 1)[1]
                copied = run_dir / relative
                original = Path(value)
                if (copied.resolve().is_relative_to(run_dir.resolve()) and copied.is_file()
                        and original.is_file() and file_hash(copied) == file_hash(original)):
                    return str(copied.resolve())
            return value
        # Raw ASR JSON and content checkpoints are immutable cache artifacts.
        # Only delivery metadata contains links that should follow the copy.
        for name in ("metadata.json", "postprocess.audit.json"):
            for path in (run_dir / "items").rglob(name):
                write_json(path, relocate(json.loads(path.read_text())))
        effective = config.to_dict()
        effective.setdefault("runtime", {}).update(resume_verified=True,
            resume_parent_dir=str(old.resolve()), resume_run_dir=str(run_dir.resolve()),
            freeze_source_context=bool(snapshot["selected_sources_captured"]),
            frozen_source_contexts={name: digest for name, digest in snapshot["files"].items()
                                    if name.endswith("/source.context.json")}, frozen_media=snapshot.get("media", {}))
        config = NextEchoConfig(effective)
    def check(stage):
        mark_pipeline_stage(run_dir, stage)
        if cancel():
            raise PipelineError("CANCELLED", "任务已取消", stage=stage)
        progress(stage=stage)
    check("resolve")
    signature = value_hash({"sources": req.sources, "limit": req.limit})
    resolved_path = run_dir / "resolved_sources.json"
    old_resolved = Path(resume_from) / "resolved_sources.json" if resume_from else None
    if old_resolved and old_resolved.is_file():
        saved = json.loads(old_resolved.read_text())
        if saved.get("source_signature") != signature or saved.get("output_signature") != value_hash(saved.get("descriptors")):
            raise PipelineError("CACHE_MISMATCH", "来源选择记录已损坏，不能在恢复时改选最新节目", blocked=True)
        descriptors = [SourceDescriptor(**r) for r in saved["descriptors"]]
    else:
        descriptors = resolve_media_inputs(list(req.sources), limit=req.limit)
    # Different sources may resolve to the same label; preserve all items.
    for index, descriptor in enumerate(descriptors, 1):
        descriptor.source_label = f"{index:03d}_" + descriptor.source_label.split("_", 1)[-1]
    selected = [d.to_dict() for d in descriptors]
    write_json(resolved_path, {"schema_version": 1, "source_signature": signature,
                "output_signature": value_hash(selected), "descriptors": selected})
    chosen = config.get_path("resolved", "asr_engine")
    settings = config.get_path("models", plan.asr_profile, default={})
    model = config.get_path("resolved", "asr_model")
    if chosen in {"sensevoice", "fun-asr-nano"}:
        model = req.asr_model or settings.get("path") or model
    if chosen == "whisper-cpp":
        from .transcription import DEFAULT_MODEL_ROOT, KNOWN_MODEL_ROOTS
        from .whisper_models import cached_model
        cached = cached_model(model, [DEFAULT_MODEL_ROOT, *KNOWN_MODEL_ROOTS])
        if cached is None and not req.allow_fallback:
            raise PipelineError("MODEL_MISSING", "Whisper权重未准备，普通run不会自动下载", blocked=True)
        model = str(cached.resolve()) if cached else model
    with ExecutionLease(on_wait=lambda: check("queued")):
        check("asr")
        payload = transcribe_media_sources(descriptors, output_dir=run_dir, model_name=model if chosen == "whisper-cpp" else "large-v3-turbo-q5_0",
                    asr_engine=chosen, asr_model=model, language=req.language, max_seconds=req.max_seconds,
                    enable_enhancement=False, postprocess_profile="off", config=config, defer_delivery=True,
                    progress_callback=lambda event: check(str(event.get("stage", "running"))))
        results = payload["results"]
        if req.max_seconds:
            for item in results:
                item.setdefault("asr", {}).setdefault("coverage", {}).update(partial=True, max_seconds=req.max_seconds)
                raw = Path(item["json_path"]) if item.get("json_path") else None
                if raw and raw.is_file():
                    data = json.loads(raw.read_text())
                    data.setdefault("coverage", {}).update(partial=True, max_seconds=req.max_seconds)
                    write_json(raw, data)
                    if item.get("cache_artifact_hashes") is not None:
                        item["cache_artifact_hashes"][raw.name] = file_hash(raw)
                if item.get("output_dir"):
                    write_json(Path(item["output_dir"]) / "metadata.json", item)
        if results and all(r.get("status") == "failed" for r in results):
            cause = next((r for r in results if r.get("blocked")), results[0])
            raise PipelineError(cause.get("error_code") or "ASR_FAILED", cause.get("error") or "所有输入均未完成识别",
                                blocked=bool(cause.get("blocked")), stage="asr")
        if req.timestamps == "precise":
            raise PipelineError("NEEDS_ALIGNER", "当前档案尚未验收精确时间轴，已保留文字；请使用segment模式", blocked=True, stage="align")
        def speaker_stage(_payload):
            from .speaker_service import build_run_speakers
            check("speaker")
            view = build_run_speakers(run_dir,mode=req.speaker_mode,config=config,
                progress_callback=lambda event: check(str(event.get("stage") or "speaker")))
            saved = json.loads((run_dir / "manifest.json").read_text())["results"]
            if len(saved) != len(results):
                raise PipelineError("PROTOCOL_ERROR", "speaker阶段改变了来源条目数")
            for item, persisted in zip(results,saved):
                item.update(persisted)
            return view
        def polish_stage():
            if req.postprocess != "off" and any(r.get("status") in {"ready", "cached"} and r.get("char_count", 0) for r in results):
                check("polish")
                llm = PROFILES[plan.polish_profile]
                provider = config.get_path("providers", "splash" if llm.engine == "local-chat-compatible" else "ollama", default={})
                endpoint = provider.get("endpoint", "http://127.0.0.1:8081" if llm.engine == "local-chat-compatible" else "http://127.0.0.1:11434")
                from .owned_provider import owned_provider
                with owned_provider(provider, endpoint=endpoint, model=llm.model, config=config) as service:
                    # External resident model requests still need a measured KV budget.
                    tags = request_json(endpoint, "/v1/models" if llm.engine == "local-chat-compatible" else "/api/tags", timeout=3)
                    available = [r.get("id") for r in tags.get("data", [])] if llm.engine == "local-chat-compatible" else [r.get("name") for r in tags.get("models", [])]
                    if llm.model not in available:
                        raise PipelineError("MODEL_MISSING", "所选本地润色模型未就绪，未切换模型", blocked=True)
                    estimate = int(provider.get("request_estimated_bytes", 128 * 1024**2) if service else provider.get("estimated_bytes", 128 * 1024**2))
                    if llm.engine == "ollama":
                        resident = request_json(endpoint, "/api/ps", timeout=3)
                        if llm.model not in [r.get("name") for r in resident.get("models", [])]:
                            size = next((int(r.get("size", 0)) for r in tags.get("models", []) if r.get("name") == llm.model), 0)
                            if not size:
                                raise PipelineError("RESOURCE_BLOCKED", "未取得非驻留模型的载入预算", blocked=True)
                            estimate += int(size * 1.5)
                    model_policy(estimate, 1, float(provider.get("timeout_seconds", 120)), config).preflight(sample_resources())
                    effective = config.to_dict()
                    context_cap = int(config.get_path("postprocess", "context_tokens", default=16384))
                    if llm.engine == "local-chat-compatible":
                        detail = next((r for r in tags.get("data", []) if r.get("id") == llm.model), {})
                        server_cap = detail.get("max_model_len") or detail.get("context_length")
                        if isinstance(server_cap, int) and server_cap > 0:
                            context_cap = min(context_cap, server_cap)
                    effective.setdefault("postprocess", {}).update(context_tokens=context_cap, provider=llm.engine, endpoint=endpoint,
                            preferred_model_id=llm.model, fallback_model=llm.model, quality_model=llm.model,
                            ownership="external", strict_selection=True, request_timeout_seconds=provider.get("timeout_seconds", 120))
                    effective["postprocess"]["service_version"] = provider.get("service_version", "unknown")
                    from .postprocess import process_transcript_artifacts
                    for item in results:
                        if item.get("status") not in {"ready", "cached"} or not item.get("char_count"):
                            continue
                        check("polish")
                        from .text_inputs import current_speaker_inputs
                        audit = process_transcript_artifacts(Path(item["output_dir"]), profile=req.profile,
                                config=NextEchoConfig(effective), summarize=req.postprocess == "polish-and-summary",
                                input_artifacts=current_speaker_inputs(item, enabled=req.speaker_mode != "off"))
                        if service:
                            audit["owned_service"] = service
                        item["enhancement"] = audit
                        item.update(audit.get("paths") or {})
                        write_json(Path(item["output_dir"]) / "metadata.json", item)
        run_post_asr_stages(payload, speaker=speaker_stage, polish=polish_stage, speaker_mode=req.speaker_mode,
                            native_speaker_alignment=True)
        check("package")
        return publish_pipeline(run_dir, job_id=job_id, results=results, plan=plan.to_dict(),
            stages={"asr":"completed", "speaker":"off" if req.speaker_mode == "off" else "completed",
                    "polish":"off" if req.postprocess == "off" else "completed"}, elapsed_seconds=time.monotonic()-started)
