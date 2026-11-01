"""Bounded media commands; arguments and raw metadata stay in private scratch."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

from .contracts import PipelineError
from .io_utils import redact, write_json
from .job_store import current_cancelled
from .resources import (
    GIB,
    ResourcePolicy,
    ExecutionLease,
    model_policy,
    private_dir,
    run_owned,
    runtime_dir,
    sample_resources,
)


def run_media_command(
    command: list[str],
    *,
    timeout: float = 120,
    stage: str = "media",
    check: bool = True,
    capture_output: bool = True,
    text: bool = True,
    input: str | None = None,
    config=None,
    max_output_bytes: int = 8 * 1024**2,
    resource_policy=None,
):
    """CompletedProcess-compatible stdout without unbounded pipe buffering.

    POSIX uses the shared lease and owned guardian. Legacy non-POSIX commands
    retain a finite subprocess timeout; that path is not a headless capability.
    """
    try:
        timeout = float(timeout)
        max_output_bytes = int(max_output_bytes)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PipelineError("INVALID_INPUT", "媒体命令时限或输出预算无效") from exc
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(a, str) for a in command)
        or not math.isfinite(float(timeout))
        or timeout <= 0
        or max_output_bytes <= 0
    ):
        raise PipelineError("INVALID_INPUT", "媒体命令、时限或输出预算无效")
    if os.name != "posix":
        if resource_policy is not None:
            raise PipelineError(
                "PLATFORM_UNSUPPORTED",
                "自定义模型命令的所属组资源守护需要POSIX运行时",
                stage=stage,
                blocked=True,
            )
        result = subprocess.run(
            command,
            capture_output=capture_output,
            text=text,
            input=input,
            timeout=timeout,
            check=check,
        )
        return result
    if current_cancelled():
        raise PipelineError("CANCELLED", "任务已取消", stage=stage)

    def check_queued_cancel():
        if current_cancelled():
            raise PipelineError("CANCELLED", "排队中的媒体任务已取消", stage=stage)

    logs = private_dir(runtime_dir() / "media-logs")
    log = logs / f"{uuid4().hex}.log"
    if resource_policy is not None and not isinstance(resource_policy, ResourcePolicy):
        raise PipelineError(
            "CONFIG_ERROR", "命令资源预算类型无效", stage=stage, blocked=True
        )
    policy = resource_policy or model_policy(64 * 1024**2, 2 * GIB, timeout, config)
    if policy.reserve_bytes < 3 * GIB or policy.timeout_seconds > timeout:
        raise PipelineError(
            "CONFIG_ERROR",
            "命令资源预算不得降低3GiB预留或绕过截止时间",
            stage=stage,
            blocked=True,
        )
    with (
        ExecutionLease(on_wait=check_queued_cancel),
        tempfile.TemporaryDirectory(
            prefix="media-", dir=private_dir(runtime_dir() / "requests")
        ) as scratch,
    ):
        root = Path(scratch)
        stdout, stderr, response = (
            root / "stdout",
            root / "stderr",
            root / "response.json",
        )
        request = {
            "command": command,
            "cwd": os.getcwd(),
            "stdout_path": str(stdout),
            "stderr_path": str(stderr),
            "response_path": str(response),
            "max_output_bytes": max_output_bytes,
        }
        if input is not None:
            body = input.encode("utf-8")
            if len(body) > max_output_bytes:
                raise PipelineError(
                    "INVALID_INPUT", "媒体命令输入超过大小上限", stage=stage
                )
            stdin = root / "stdin"
            stdin.write_bytes(body)
            request["stdin_path"] = str(stdin)
        request_path = root / "request.json"
        write_json(request_path, request)
        resources = run_owned(
            [
                sys.executable,
                str(Path(__file__).with_name("media_worker.py")),
                str(request_path),
            ],
            log_path=log,
            policy=policy,
            cancel=current_cancelled,
            sampler=sample_resources,
            interval=0.2,
        )
        import json

        if not response.is_file() or response.stat().st_size > 4096:
            raise PipelineError("PROTOCOL_ERROR", "媒体命令回执缺失或过大", stage=stage)
        data = json.loads(response.read_text())
        write_json(
            log.with_suffix(".diagnostic.json"),
            {
                "stage": stage,
                "binary": Path(command[0]).name,
                "effective_reserve_bytes": policy.reserve_bytes,
                "output_bytes": data.get("output_bytes"),
                "returncode": data.get("returncode"),
                "error_code": data.get("error_code"),
                "resources": resources,
            },
        )
        if data.get("error_type"):
            if data["error_type"] == "FileNotFoundError":
                raise FileNotFoundError(data.get("errno"), "媒体程序不存在", command[0])
            raise OSError(data.get("errno"), "媒体程序无法启动")
        if data.get("error_code"):
            raise PipelineError(
                data["error_code"],
                "媒体命令输出超过大小上限，详情见本机media-logs",
                stage=stage,
            )
        if sum(p.stat().st_size for p in (stdout, stderr)) > max_output_bytes:
            raise PipelineError("OUTPUT_LIMIT", "媒体命令输出超过大小上限", stage=stage)
        result = subprocess.CompletedProcess(
            command,
            int(data["returncode"]),
            stdout.read_text(encoding="utf-8", errors="replace")
            if text
            else stdout.read_bytes(),
            stderr.read_text(encoding="utf-8", errors="replace")
            if text
            else stderr.read_bytes(),
        )
        if check and result.returncode:
            raise subprocess.CalledProcessError(
                result.returncode,
                redact(" ".join(command)),
                output=redact(result.stdout) if text else result.stdout,
                stderr=redact(result.stderr) if text else result.stderr,
            )
        return result
