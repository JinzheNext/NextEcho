"""File delivery and bounded machine output, independent of model runtimes."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .contracts import PipelineError
from .io_utils import atomic_write, file_hash, redact, redact_structure, value_hash, write_json
from .reading_artifacts import freeze_delivery_context, markdown_text, metadata_header, primary_path, reading_filename, render_comments
from urllib.parse import quote

EXIT_CODES = {"ready": 0, "partial": 4, "failed": 3, "blocked": 5, "cancelled": 6}
TERMINAL = frozenset(EXIT_CODES)
ERROR_ACTIONS = {
    "OWNERSHIP_CONFLICT": "更改专用服务端口，或显式选择已有external服务；不要停止不属于本任务的程序。",
    "CONFIG_ERROR": "检查所选配置文件及安装绑定，修正后重试。",
    "MODEL_CANDIDATE": "先完成模型资源与质量验收；候选试用需显式指定--profile。",
    "MODEL_MISSING": "运行nextecho prepare绑定所选模型和VAD文件。",
    "MODEL_INVALID": "所选模型或资产台账无效；核对回执标明的模型、文件及内容hash后重试。",
    "MODEL_BUSY": "模型下载或本地执行正忙；稍后重试，不重复启动下载。",
    "MODEL_LOCK_LEGACY": "确认没有旧下载进程后手动移走模型目录中的旧锁；不要删除活动锁。",
    "MODEL_LOCK_INVALID": "核对模型目录锁文件权限和类型，保留活动下载文件。",
    "MODEL_DOWNLOAD_FAILED": "固定版本模型下载失败，旧文件保留；检查网络后重试download-model。",
    "REQUEST_NOT_FOUND": "该意图尚未提交；继续用已保存的请求文件提交，不要另建UUID。",
    "DEPENDENCY_MISSING": "在所选worker环境安装对应ASR可选依赖，然后重新prepare。",
    "RESOURCE_BLOCKED": "检查calibration.json及可用内存，资源满足后用jobs resume恢复。",
    "RESOURCE_LIMIT": "检查资源曲线，释放内存后恢复，不要停止不属于此任务的服务。",
    "PROVIDER_UNAVAILABLE": "检查已配置的本机模型服务和端口。",
    "PROVIDER_BUSY_UNKNOWN": "先确认旧请求在服务端已结束，再解除占用。",
    "NEEDS_ALIGNER": "使用segment模式，或先验收精确时间轴。",
    "CACHE_MISMATCH": "使用新输出目录，或对相同计划的中断任务执行jobs resume。",
    "INTERRUPTED": "用jobs resume创建新attempt，复用已校验阶段。",
    "SOURCE_AUTH_REQUIRED": "配置用户提供的cookie文件路径后重试，不要把cookie内容贴入对话。",
    "SOURCE_UNAVAILABLE": "检查文件是否存在、链接是否仍有效以及媒体是否可访问，再重试。",
    "STAGE_TIMEOUT": "检查本机阶段日志及资源曲线；确认所属工作已结束后调整时限并恢复。",
    "WORKER_FAILED": "查看对应items目录下logs中的worker日志，修复运行环境后恢复。",
    "ASR_FAILED": "查看receipt中各输入的错误及items下的识别日志，修复后恢复。",
    "PROTOCOL_ERROR": "检查本地worker日志和运行时版本；不要把损坏回执当成成功结果。",
}


def compact(receipt: dict) -> dict:
    result = {key: receipt.get(key, "") for key in
              ("schema_version", "job_id", "status", "primary_file", "receipt_path")}
    result["schema_version"] = 1
    if receipt.get("error_code"):
        result["error_code"] = receipt["error_code"]
        result["message"] = redact(str(receipt.get("message", "")))[:160]
        result["next_action"] = receipt.get("next_action") or ERROR_ACTIONS.get(receipt["error_code"], "查看本地诊断文件。")
    if len(json.dumps(result, ensure_ascii=False).encode()) > 2048:
        raise PipelineError("OUTPUT_PATH_TOO_LONG", "结果路径超过回执预算，请使用较短的输出目录")
    return result


def assemble(run_dir: Path, results: list[dict]) -> Path:
    context = freeze_delivery_context(run_dir)
    rows = ["# NextEcho，让播客自动本地转录进知识库", "",
            "@金哲Next，一起用好AI走向Next Level，欢迎关注小红书&公众号～", ""]
    for index, result in enumerate(results, 1):
        rows += [f"## {index}. {markdown_text(result.get('title') or result.get('source_label') or '媒体')}", ""]
        rows += metadata_header(result, context, item_index=index)
        if result.get("error"):
            rows += [markdown_text(redact(str(result["error"]))), ""]
        path = result.get("identity_text_path") or result.get("enhanced_text_path") or result.get("dialogue_text_path") or result.get("speaker_text_path") or result.get("normalized_text_path") or result.get("text_path")
        enhanced = result.get("identity_json_path") or result.get("enhanced_json_path")
        enhanced_data = None
        layer = "raw" if result.get("identity_json_path") else "enhanced"
        if enhanced and Path(enhanced).is_file():
            candidate = json.loads(Path(enhanced).read_text(encoding="utf-8"))
            field = "transcription" if layer == "raw" else "enhanced_transcription"
            if isinstance(candidate.get(field), list) and candidate[field]:
                enhanced_data = candidate
        if enhanced_data is not None:
            from .postprocess import load_segments, segment_to_dict
            from .reading_view import render_reading
            source = Path(enhanced).resolve()
            source.relative_to(run_dir.resolve())
            segments = load_segments(enhanced_data, layer=layer)
            body, audit = render_reading([segment_to_dict(s) for s in segments], result.get("source_context") or {}, item_index=index)
            reading_path = source.parent / "transcript.reading.md"
            audit_path = source.parent / "reading.audit.json"
            atomic_write(reading_path, body)
            audit.update(source_path=str(source), source_file_sha256=file_hash(source), reading_sha256=file_hash(reading_path))
            write_json(audit_path, audit)
            result.update(reading_markdown_path=str(reading_path), reading_audit_path=str(audit_path))
            rows += [body, ""]
        elif path and Path(path).is_file():
            resolved = Path(path).resolve()
            resolved.relative_to(run_dir.resolve())
            rows += [markdown_text(resolved.read_text(encoding="utf-8")), ""]
        coverage = (result.get("asr") or {}).get("coverage") or {}
        if coverage.get("content_state") == "no_speech":
            rows += ["没有检测到可识别语音。", ""]
        if coverage.get("partial"):
            rows += ["本项仅覆盖部分音频。", ""]
        comments = render_comments(result.get("source_context") or {}, item_index=index)
        if comments:
            rows += [comments, ""]
            # The main document includes comments, so importing only that file
            # into an Obsidian vault does not leave a broken companion link.
            item_source = result.get("enhanced_json_path") or result.get("text_path")
            if item_source and Path(item_source).is_file():
                folder = Path(item_source).resolve().parent
                folder.relative_to(run_dir.resolve())
                comments_path = folder / "source.comments.md"
                atomic_write(comments_path, comments)
                result["comments_markdown_path"] = str(comments_path)
    rows += ["powered by github.com/JinzheNext/NextEcho - 让播客自动转录进知识库", ""]
    path = run_dir / reading_filename(results, context["transcribed_on"])
    body = "\n".join(rows)
    atomic_write(path, body)
    atomic_write(run_dir / "result.md", body)
    write_json(run_dir / "delivery.index.json", {"schema_version": 1, "primary_name": path.name,
               "primary_sha256": file_hash(path), "legacy_alias": "result.md",
               "transcribed_on": context["transcribed_on"]})
    return path


def write_receipt(run_dir: Path, *, job_id: str, status: str, results: list[dict] | None = None,
                  error_code: str = "", message: str = "", plan: dict | None = None, stage: str = "complete") -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    primary = primary_path(run_dir)
    payload = {"schema_version": 1, "job_id": job_id, "status": status,
               "primary_file": str(primary.resolve()) if primary.is_file() else "",
               "receipt_path": str((run_dir / "receipt.json").resolve()),
               "error_code": error_code, "message": redact(message),
               "completed_at": datetime.now(UTC).isoformat(), "plan": redact_structure(plan), "items": redact_structure(results or [])}
    payload["stage"] = stage
    payload["completion"] = [completion_details(item) for item in results or []]
    payload["plan_fingerprint"] = value_hash(plan)
    payload["content_state"] = "no_speech" if results and all((r.get("asr") or {}).get("coverage", {}).get("content_state") == "no_speech" for r in results) else "speech" if results else "unknown"
    payload["next_action"] = ERROR_ACTIONS.get(error_code, "查看本地诊断文件。") if error_code else ""
    payload["retryable"] = error_code in {"RESOURCE_BLOCKED", "PROVIDER_UNAVAILABLE", "INTERRUPTED", "SOURCE_AUTH_REQUIRED"}
    artifacts = [primary, run_dir / "result.md", run_dir / "delivery.context.json", run_dir / "delivery.index.json",
                 run_dir / "manifest.json", run_dir / "run_config.json", run_dir / "resolved_sources.json",
                 run_dir / "resume.plan.json", run_dir / "resume.inputs.json"]
    def collect(value, *, required=status in {"ready", "partial"}):
        if isinstance(value, dict):
            if value.get("status") in {"failed", "blocked", "cancelled"}:
                required = False
            for key, child in value.items():
                if key.endswith("_path") and isinstance(child, str) and child:
                    path = Path(child)
                    if path.is_absolute() and path.resolve().is_relative_to(run_dir.resolve()):
                        if required and not path.is_file():
                            raise PipelineError("PROTOCOL_ERROR", "交付产物缺失：" + path.name)
                        if path.is_file():
                            artifacts.append(path)
                elif isinstance(child, (dict, list)):
                    collect(child, required=required)
        elif isinstance(value, list):
            for child in value:
                collect(child, required=required)
    collect(results or [])
    if status in {"ready", "partial"} and not primary.is_file():
        raise PipelineError("PROTOCOL_ERROR", "交付主文件尚未生成")
    payload["artifact_hashes"] = {str(p.resolve()): file_hash(p) for p in artifacts if p.is_file()}
    compact(payload)  # Validate BEFORE publishing a success receipt.
    write_json(run_dir / "receipt.json", payload)
    if error_code or status == "partial":
        write_json(run_dir / "errors.json", {"generated_at": payload["completed_at"],
                   "error_code": error_code, "message": redact(message),
                   "items": redact_structure([r for r in results or [] if r.get("status") == "failed" or (r.get("enhancement") or {}).get("status") == "partial"])})
    return payload


def completion_details(item: dict) -> dict:
    """Keep full audio coverage, model adoption and speaker review independent."""
    coverage = (item.get("asr") or {}).get("coverage") or {}
    enhancement = item.get("enhancement") or {}
    polish = enhancement.get("polish") or {}
    return {"source_label": item.get("source_label", ""),
            "audio": {"full": coverage.get("partial") is False,
                      "covered_seconds": coverage.get("covered_seconds"),
                      "duration_seconds": item.get("duration_seconds")},
            "polish": {"status": enhancement.get("status", "not_requested"),
                       "processing_complete": polish.get("processing_complete", False),
                       "chunks": polish.get("chunks"),
                       "original_fallback_chunks": len(polish.get("original_fallback_chunks", [])),
                       "retained_source_segments": len(polish.get("retained_source_segments", [])),
                       "unprocessed_segments": len(polish.get("unprocessed_segment_ids", [])),
                       "review_segments": len(polish.get("review_segments", [])),
                       "errors": polish.get("errors", [])},
            "speakers": {"status": item.get("speaker_status", "not_requested"),
                         "identity_review_independent_of_audio_coverage": True}}
