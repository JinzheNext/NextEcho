"""Translate a runtime's bounded receipt to a verified path or compact JSON."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from bind_runtime import binding_path
from request_intent import create_intent, has_key, load_intent, mutates, validate_binding


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--binding", type=Path, default=binding_path())
    p.add_argument("--output", choices=["paths", "json"], default="paths")
    p.add_argument("--create-request", action="store_true", help="仅持久化本次意图，不提交；保存返回的request_file以便重试")
    p.add_argument("--request-file", type=Path, help="执行已保存意图；不得附加改变原请求的参数")
    a, arguments = p.parse_known_args()
    request_path = None
    try:
        if a.request_file and not a.create_request:
            if arguments:
                raise ValueError("执行已有请求文件时不能追加参数；新要求请创建新请求")
            request_path, intent = load_intent(a.request_file)
            binding, arguments = intent["binding"], intent["arguments"]
        else:
            binding = json.loads(a.binding.expanduser().read_text())
            validate_binding(binding)
            if not arguments:
                p.error("需要媒体输入，或jobs命令")
            intent = None
            if mutates(arguments):
                request_path, intent = create_intent(binding, arguments, a.binding, a.request_file)
            if a.create_request:
                if intent is None:
                    raise ValueError("查询/等待/取消不需要创建请求")
                result = {"schema_version": 1, "status": "request_prepared", "request_id": intent["request_id"],
                          "request_file": str(request_path)}
                if len(json.dumps(result, ensure_ascii=False).encode()) > 2048:
                    raise ValueError("请求文件路径过长；请使用较短目录")
                print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
                return 0
        validate_binding(binding)
        command = arguments if arguments[0] in {"jobs", "refine-speakers"} else ["run", *arguments]
        command = list(command)
        if intent is not None and not has_key(command):
            command += ["--idempotency-key", "intent-" + intent["request_id"]]
        if arguments[0] not in {"jobs", "refine-speakers"} and not any(arg == "--speaker-mode" or arg.startswith("--speaker-mode=") for arg in arguments):
            command += ["--speaker-mode", "auto"]
        command += ["--output", "json"]
        if binding.get("config"):
            command = ["--config", binding["config"], *command]
        env = dict(os.environ)
        if binding.get("runtime_dir"):
            env["NEXTECHO_RUNTIME_DIR"] = binding["runtime_dir"]
        def invoke(selected):
            return subprocess.run([binding["python"], "-m", "workbench.cli", *selected],
                                  capture_output=True, text=True, env=env,
                                  cwd=intent["caller_cwd"] if intent is not None else None)
        completed = None
        if a.request_file and not has_key(arguments):
            # The saved UUID belongs to this immutable intent. Resolve an
            # already-accepted request BEFORE re-reading changed config.
            lookup = invoke(["jobs", "lookup", "intent-" + intent["request_id"], "--output", "json"])
            if len(lookup.stdout.encode()) > 2048:
                raise ValueError("PROTOCOL_ERROR: 查询回执超过2KiB")
            found = json.loads(lookup.stdout)
            if found.get("error_code") != "REQUEST_NOT_FOUND":
                if found.get("job_id") and found.get("status") in {"queued", "running"} and arguments[0] != "jobs" and "--detach" not in arguments:
                    completed = invoke(["jobs", "wait", found["job_id"], "--timeout", "60", "--output", "json"])
                else:
                    completed = lookup
        if completed is None:
            completed = invoke(command)
        if len(completed.stdout.encode()) > 2048:
            raise ValueError("PROTOCOL_ERROR: 回执超过2KiB，请检查运行时版本；完整信息留在本地任务目录")
        result = json.loads(completed.stdout)
        if result.get("schema_version") != 1 or not result.get("status"):
            raise ValueError("PROTOCOL_ERROR: 无效回执")
        expected = {"ready": 0, "partial": 4, "failed": 3, "blocked": 5, "cancelled": 6}
        if result["status"] not in {*expected, "queued", "running"}:
            raise ValueError("PROTOCOL_ERROR: 未知任务状态")
        if result["status"] in {"queued", "running"} and completed.returncode not in {0, 7}:
            raise ValueError("PROTOCOL_ERROR: 等待状态与退出码不一致")
        if not result.get("job_id") and result.get("error_code") == "INVALID_INPUT":
            expected["failed"] = 2
        if result["status"] in expected and completed.returncode != expected[result["status"]]:
            raise ValueError("PROTOCOL_ERROR: 结果状态与退出码不一致")
        if result["status"] in {"ready", "partial"}:
            primary = Path(result.get("primary_file", ""))
            if not primary.is_absolute() or not primary.is_file():
                raise ValueError("PROTOCOL_ERROR: 结果文件不存在")
        if a.output == "json" or result["status"] not in {"ready", "partial"}:
            print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        else:
            if result.get("artifact_kind") == "speaker-refinement":
                delivered = [*result.get("speaker_markdown_paths", []), *result.get("speaker_identity_review_paths", [])]
                for path in delivered:
                    if path and Path(path).is_absolute() and Path(path).is_file():
                        print(path)
            elif result.get("primary_file"):
                print(result["primary_file"])
            if result["status"] == "partial":
                print(f"部分完成；详情：{result.get('receipt_path', '')}", file=sys.stderr)
        return completed.returncode
    except (OSError, ValueError, KeyError, TypeError) as exc:
        retry = f" 重试请使用 --request-file {request_path}" if request_path is not None else ""
        print(f"NextEcho: {str(exc)[:240]}。未绑定时请按 references/install.md 安装。{retry}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
