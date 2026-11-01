"""Agent-facing commands: bounded receipts, no transcript on stdout."""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

from .config import load_config
from .contracts import PipelineError, PipelineRequest, plan_request
from .delivery import EXIT_CODES, TERMINAL, compact
from .io_utils import redact, write_json
from .job_store import JobStore
from .model_profiles import PROFILES, validate_model_files
from .resources import private_dir, process_alive, runtime_dir
from .sources import absolute_media_inputs


def bound_config(config_path=None):
    return load_config(config_path, strict=True)


def add_commands(subparsers):
    run = subparsers.add_parser("run", help="无网页转写，默认只返回结果文件路径")
    run.add_argument("sources", nargs="+")
    run.add_argument("--config", dest="config_path", default=argparse.SUPPRESS)
    run.add_argument("--profile", choices=["fast", "balanced", "quality"])
    run.add_argument("--postprocess", choices=["off", "polish", "polish-and-summary"])
    run.add_argument("--speaker-mode", choices=["off", "auto", "required"])
    run.add_argument("--timestamps", choices=["segment", "precise"], default="segment")
    run.add_argument("--language")
    run.add_argument("--max-seconds", type=int, default=0)
    run.add_argument("--limit", type=int, default=1)
    run.add_argument("--asr-engine", choices=["fun-asr-nano", "sensevoice", "qwen3-asr", "whisper-cpp"], default="")
    run.add_argument("--asr-model", default="")
    run.add_argument("--polish-model", choices=[p.id for p in PROFILES.values() if p.task == "polish"], default="")
    run.add_argument("--allow-fallback", action="store_true")
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--output", choices=["paths", "json"], default="paths")
    run.add_argument("--detach", action="store_true")
    run.add_argument("--idempotency-key")
    run.set_defaults(func=cmd_run)
    jobs = subparsers.add_parser("jobs", help="查询、等待、取消或恢复本地任务")
    actions = jobs.add_subparsers(dest="action", required=True)
    for name in ("status", "wait", "cancel", "resume"):
        p = actions.add_parser(name)
        p.add_argument("job_id")
        p.add_argument("--output", choices=["paths", "json"], default="json")
        if name == "wait":
            p.add_argument("--timeout", type=float, default=60)
        if name == "resume":
            p.add_argument("--mode", choices=["exact", "refresh"], default="exact")
            p.add_argument("--repair-polish", action="store_true", help="refresh时按句重验历史模型响应，只重试未通过的句子")
            p.add_argument("--config", dest="config_path", default=argparse.SUPPRESS)
            p.add_argument("--idempotency-key")
        p.set_defaults(func=cmd_jobs)
    idle = actions.add_parser("provider-idle", help="仅在用户确认服务已空闲后解除不确定的请求占用")
    idle.add_argument("endpoint")
    idle.add_argument("--confirm", action="store_true", required=True)
    idle.set_defaults(func=cmd_jobs, output="json")
    lookup = actions.add_parser("lookup", help="按持久请求意图键查询，响应丢失时不重复提交")
    lookup.add_argument("idempotency_key")
    lookup.add_argument("--output", choices=["paths", "json"], default="json")
    lookup.set_defaults(func=cmd_jobs)
    p = subparsers.add_parser("prepare", help="绑定本地模型和独立环境，不启动网页")
    p.add_argument("--config", dest="config_path", default=argparse.SUPPRESS)
    p.add_argument("--profile", choices=["fast", "balanced", "quality"], default="balanced")
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--worker-python", type=Path, default=Path(sys.executable))
    p.add_argument("--vad-model", type=Path)
    p.add_argument("--speaker-model-root", type=Path, help="可选的segmentation.onnx/embedding.onnx所在目录")
    p.add_argument("--calibrate-audio", type=Path, help="≤30秒16kHz单声道WAV，受控测量资源后仍保持candidate")
    p.add_argument("--no-web", action="store_true")
    p.set_defaults(func=cmd_prepare)


def emit(job, mode="json", *, accepted=False):
    result = compact(job)
    if mode == "json" or accepted:
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    else:
        if result.get("primary_file"):
            print(result["primary_file"])
        if result["status"] != "ready":
            print(f"{result['status']}: {result.get('error_code', '')} {result.get('message', '')} {result['receipt_path']}", file=sys.stderr)
    return EXIT_CODES.get(job["status"], 0)


def error(exc, mode="json", receipt_path=""):
    code = getattr(exc, "code", "INVALID_INPUT")
    result = emit({"schema_version": 1, "job_id": "", "status": "blocked" if getattr(exc, "blocked", False) else "failed",
                 "primary_file": "", "receipt_path": receipt_path, "error_code": getattr(exc, "code", "INVALID_INPUT"),
                 "message": redact(str(exc))[:160]}, mode)
    return 2 if code == "INVALID_INPUT" else result


def wait_for(store, job_id, seconds=None):
    until = None if seconds is None else time.monotonic() + seconds
    while True:
        job = store.get(job_id)
        if job["status"] in TERMINAL or until is not None and time.monotonic() >= until:
            return job
        time.sleep(.2)


def approved_trial(config, model_id):
    """Local opt-in for an exact model ID; never a quality promotion."""
    approved = config.get_path("agent", "approved_trial_models", default=[])
    if not isinstance(approved, list) or any(not isinstance(item, str) or item not in PROFILES for item in approved):
        raise PipelineError("CONFIG_ERROR", "agent.approved_trial_models须为已登记模型ID列表", blocked=True)
    return model_id in approved


def cmd_run(args):
    job = None
    try:
        if os.name != "posix":
            raise PipelineError("PLATFORM_UNSUPPORTED", "此平台请使用原兼容命令", blocked=True)
        cfg = bound_config(getattr(args, "config_path", None))
        submission_context = {"caller_cwd": str(Path.cwd()), "original_inputs": list(args.sources)}
        request = PipelineRequest(tuple(absolute_media_inputs(args.sources)), profile=args.profile or cfg.get_path("agent", "default_profile", default="balanced"),
            postprocess=args.postprocess or cfg.get_path("agent", "postprocess", default="polish"),
            speaker_mode=args.speaker_mode or cfg.get_path("agent", "speaker_mode", default="off"), timestamps=args.timestamps,
            language=args.language or cfg.get_path("asr", "language", default="zh"), max_seconds=args.max_seconds,
            limit=args.limit, asr_model=args.asr_model, asr_engine=args.asr_engine, polish_model=args.polish_model,
            allow_fallback=args.allow_fallback)
        plan = plan_request(request, cfg)
        if (args.profile is None and not args.asr_engine
                and cfg.get_path("models", plan.asr_profile, "release_status", default="candidate") != "ready"
                and not approved_trial(cfg, plan.asr_profile)):
            raise PipelineError("MODEL_CANDIDATE", "默认识别档案尚未通过产品验收；如需受控试用，请显式选择--profile", blocked=True)
        if (args.profile is None and not args.polish_model and request.postprocess != "off"
                and cfg.get_path("models", plan.polish_profile, "release_status", default="candidate") != "ready"
                and not approved_trial(cfg, plan.polish_profile)):
            raise PipelineError("MODEL_CANDIDATE", "默认润色档案尚未通过产品验收；受控试用请显式选择--profile或--polish-model", blocked=True)
        store = JobStore()
        job, created = store.submit(plan.to_dict(), output_dir=args.output_dir, key=args.idempotency_key,
                                    submission_context=submission_context)
        if created:
            try:
                store.spawn(job["job_id"])
            except Exception:
                return emit(store.get(job["job_id"]), args.output)
        if args.detach:
            return emit(store.get(job["job_id"]), args.output, accepted=True)
        return emit(wait_for(store, job["job_id"]), args.output)
    except KeyboardInterrupt:
        print(f"等待已结束，后台任务继续：{job['job_id'] if job else ''}", file=sys.stderr)
        return 130
    except Exception as exc:
        return error(exc, args.output)


def cmd_jobs(args):
    try:
        store = JobStore()
        if args.action == "provider-idle":
            from .llm_providers import endpoint_url
            store.confirm_provider_idle(endpoint_url(args.endpoint))
            print(json.dumps({"schema_version": 1, "status": "idle_confirmed"}))
            return 0
        if args.action == "lookup":
            return emit(store.lookup_key(args.idempotency_key), args.output)
        job = store.get(args.job_id)
        if args.action == "cancel" and job["status"] not in TERMINAL:
            job = store.update(args.job_id, cancel_requested=True)
            if process_alive(job.get("owner_pid", 0), job.get("owner_birth", "")):
                os.kill(job["owner_pid"], signal.SIGTERM)
            job = wait_for(store, args.job_id, 5)
        elif args.action == "wait":
            if not 0 <= args.timeout <= 60:
                raise PipelineError("INVALID_INPUT", "等待时限必须在0至60秒")
            job = wait_for(store, args.job_id, args.timeout)
            if job["status"] not in TERMINAL:
                emit(job, args.output)
                return 7
        elif args.action == "resume":
            if job["status"] not in TERMINAL:
                raise PipelineError("INVALID_INPUT", "活动任务不能重复恢复")
            from .resume import resume_plan
            mode = getattr(args, "mode", "exact")
            plan, audit = resume_plan(job, mode=mode,
                config=bound_config(getattr(args, "config_path", None)) if mode == "refresh" else None,
                repair_polish=getattr(args, "repair_polish", False))
            job, created = store.submit(plan, parent=job["job_id"], resume_from=job["run_dir"],
                key=getattr(args, "idempotency_key", None), resume_policy=audit)
            if created:
                try:
                    store.spawn(job["job_id"])
                except PipelineError:
                    return emit(store.get(job["job_id"]), args.output)
            return emit(store.get(job["job_id"]), args.output, accepted=True)
        return emit(job, args.output)
    except Exception as exc:
        return error(exc, args.output)


def cmd_prepare(args):
    try:
        # Preserve venv symlinks: resolving bin/python selects the base
        # interpreter and silently loses the model environment's packages.
        root, python = args.model_root.expanduser().resolve(), args.worker_python.expanduser().absolute()
        if not python.is_file():
            raise PipelineError("DEPENDENCY_MISSING", "worker解释器不存在", blocked=True)
        path = private_dir(runtime_dir()) / "binding.json"
        binding = json.loads(path.read_text()) if path.is_file() else {"schema_version": 1, "config": {}}
        models = binding["config"].setdefault("models", {})
        status = {}
        for name, folder in (("fun-asr-nano-pt", "fun-asr-nano-pt"), ("sensevoice-int8", "sherpa-sense-voice")):
            settings = models.setdefault(name, {})
            settings.update(path=str(root / folder), worker_python=str(python))
            status[name] = validate_model_files(root / folder, PROFILES[name].required_files)
        vad = args.vad_model or root / "silero_vad.onnx"
        binding["config"].setdefault("asr", {})["vad_model"] = str(vad.expanduser().resolve())
        if args.speaker_model_root:
            speaker_root = args.speaker_model_root.expanduser().resolve()
            binding["config"].setdefault("diarization", {}).update(
                worker_python=str(python), ctc_model_path=str(root / "sherpa-sense-voice"),
                num_clusters=-1, sherpa={"model_paths": [str(speaker_root / name) for name in
                                                       ("segmentation.onnx", "embedding.onnx")]})
        write_json(path, binding)
        selected = "sensevoice-int8" if args.profile == "fast" else "fun-asr-nano-pt"
        if args.calibrate_audio:
            from .calibration import calibrate
            calibration_dir = path.parent / "calibrations" / f"{selected}-{int(time.time())}"
            report = calibrate(bound_config(getattr(args, "config_path", None)), selected,
                  args.calibrate_audio.expanduser().resolve(), calibration_dir)
            models[selected].update(report["settings"])
            write_json(path, binding)
        from .capabilities import inspect_capabilities
        technical = inspect_capabilities(bound_config(getattr(args, "config_path", None)), profile=args.profile, capability="asr")["capabilities"]["asr"]
        calibrated = bool(models[selected].get("estimated_bytes"))
        print(json.dumps({"schema_version": 1, "status": "candidate", "binding_path": str(path), "models": status,
            "vad_available": vad.is_file(), "resource_calibrated": calibrated, "capability": technical,
            "next_action": "绑定VAD并完成受控资源校准；候选通过G1–G7后才提升默认"}, ensure_ascii=False))
        return 0 if technical["status"] == "ready" and calibrated else 5
    except Exception as exc:
        report_path = calibration_dir / "calibration.json" if "calibration_dir" in locals() else None
        return error(exc, receipt_path=str(report_path) if report_path and report_path.is_file() else "")
