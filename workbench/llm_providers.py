"""Loopback-only local model transport, configuration scope and telemetry."""
from __future__ import annotations

import contextvars
import functools
import ipaddress
import json
import math
import re
import os
import sys
import hashlib
import tempfile
from pathlib import Path
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

from .config import load_config
from .contracts import PipelineError


@dataclass
class GenerationContext:
    settings: dict = field(default_factory=dict)
    usage: list = field(default_factory=list)
    resources: list = field(default_factory=list)


CONTEXT = contextvars.ContextVar("nextecho_generation", default=None)


def provider_configuration(function):
    """Scope configuration/telemetry, without granting service ownership.

    A settings string cannot authorize unload. Services actually started by
    owned_provider are released by that verified process owner's context.
    Direct and legacy calls only borrow the configured endpoint.
    """
    @functools.wraps(function)
    def call(*args, **kwargs):
        config = kwargs.get("config") or load_config()
        settings = dict(config.get("postprocess", {}))
        settings.setdefault("output_script", config.get("asr", {}).get("output_script", "original"))
        token = CONTEXT.set(GenerationContext(settings))
        try:
            return function(*args, **kwargs)
        finally:
            CONTEXT.reset(token)
    return call


def setting(key, default):
    context = CONTEXT.get()
    return context.settings.get(key, default) if context else default


def decode_model_json(content):
    # A single whole-message JSON fence is a serialization envelope. Never
    # search prose/thinking for an embedded object or repair malformed JSON.
    text = str(content).strip()
    envelope = "plain-json"
    fenced = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fenced:
        text, envelope = fenced.group(1), "json-fence"
    def reject_constant(token):
        raise ValueError(f"Invalid JSON constant: {token}")
    def finite_float(token):
        value = float(token)
        if not math.isfinite(value):
            raise ValueError("JSON number exceeds finite range")
        return value
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = value
        return result
    parsed = json.loads(text, parse_constant=reject_constant, parse_float=finite_float,
                        object_pairs_hook=unique_object)
    if not isinstance(parsed, dict):
        raise ValueError("Expected one JSON object")
    return parsed, envelope


def validate_json_schema(value, schema, path="$"):
    """Validate our bounded type/required/enum/properties/items schema subset."""
    kind = schema.get("type")
    valid = {"object": isinstance(value, dict), "array": isinstance(value, list),
             "string": isinstance(value, str), "integer": type(value) is int,
             "number": type(value) in {int, float}, "boolean": type(value) is bool,
             "null": value is None}
    if kind and not valid.get(kind, False):
        raise ValueError(f"Schema type mismatch at {path}: expected {kind}")
    if kind == "number" and not math.isfinite(value):
        raise ValueError(f"Schema number must be finite at {path}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"Schema enum mismatch at {path}")
    if isinstance(value, dict):
        for name in schema.get("required", []):
            if name not in value:
                raise ValueError(f"Schema required field missing at {path}.{name}")
        for name, child in value.items():
            properties = schema.get("properties", {})
            if name in properties:
                validate_json_schema(child, properties[name], f"{path}.{name}")
            elif schema.get("additionalProperties") is False:
                raise ValueError(f"Schema unexpected field at {path}.{name}")
            elif isinstance(schema.get("additionalProperties"), dict):
                validate_json_schema(child, schema["additionalProperties"], f"{path}.{name}")
    elif isinstance(value, list) and "items" in schema:
        for index, child in enumerate(value):
            validate_json_schema(child, schema["items"], f"{path}[{index}]")


def chat_model_identity(endpoint, model):
    """An alias alone cannot authorize reuse after weights change.

    Servers that expose no immutable model revision get invocation-scoped
    checkpoints. This also keeps completed blocks reusable within that call.
    """
    from uuid import uuid4
    from .io_utils import value_hash
    payload = request_json(endpoint, "/v1/models", timeout=3)
    row = next((r for r in payload.get("data", []) if r.get("id") == model), {})
    revision = row.get("digest") or row.get("model_sha256") or row.get("revision")
    return {"metadata": row, "fingerprint": value_hash(row) if revision else uuid4().hex,
            "cache_identity": "model-revision" if revision else "invocation-only-unverified"}


def endpoint_url(endpoint: str) -> str:
    parsed = urllib.parse.urlsplit(endpoint)
    if parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Only a local HTTP model endpoint is allowed")
    addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 80, type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(row[4][0]).is_loopback for row in addresses):
        raise ValueError("Model endpoint must resolve only to loopback addresses")
    return endpoint.rstrip("/")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request_json(endpoint: str, path: str, body: dict | None = None, *, timeout: float = 30) -> dict:
    endpoint = endpoint_url(endpoint)
    request = urllib.request.Request(endpoint + path,
            data=None if body is None else json.dumps(body, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    store = None
    transport_complete = False
    if body is not None and os.environ.get("NEXTECHO_JOB_ID"):
        from .job_store import JobStore
        store = JobStore()
        store.provider_start(endpoint, os.environ["NEXTECHO_JOB_ID"])
    try:
        with opener.open(request, timeout=timeout) as response:
            payload = response.read(8 * 1024 * 1024 + 1)
        if len(payload) > 8 * 1024 * 1024:
            raise ValueError("Local model response exceeds size limit")
        if store:
            store.provider_end(endpoint, completed=True)
        transport_complete = True
        return json.loads(payload)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        if store:
            refused = isinstance(getattr(exc, "reason", None), ConnectionRefusedError)
            store.provider_end(endpoint, completed=transport_complete or refused or isinstance(exc, urllib.error.HTTPError))
        raise PipelineError("PROVIDER_UNAVAILABLE", f"本地模型服务请求失败：{type(exc).__name__}", stage="polish", blocked=True) from exc


def generate_request(endpoint: str, path: str, body: dict, *, timeout: float = 120):
    """Monitor the host during external inference, retaining uncertain occupancy.

    Client RSS is reported separately; it is never used as server model RSS.
    """
    if os.name != "posix":
        return request_json(endpoint, path, body, timeout=timeout)
    from .resources import ExecutionLease, GIB, model_policy, private_dir, process_birth, run_owned, runtime_dir
    from .job_store import JobStore, current_cancelled
    from .io_utils import write_json
    endpoint = endpoint_url(endpoint)
    with ExecutionLease(), tempfile.TemporaryDirectory(prefix="provider-", dir=private_dir(runtime_dir() / "requests")) as scratch:
        if current_cancelled():
            raise PipelineError("CANCELLED", "任务已取消", stage="polish")
        model = body.get("model", "")
        # A conservative KV allowance bounds all contexts, including legacy
        # calls. Cold Ollama weights are added before the inference begins.
        context = int(setting("context_tokens", 16384))
        estimate = int(setting("estimated_bytes", max(128 * 1024**2, context * 65536)))
        if path == "/api/chat":
            resident = request_json(endpoint, "/api/ps", timeout=3)
            if model not in [r.get("name") for r in resident.get("models", [])]:
                tags = request_json(endpoint, "/api/tags", timeout=3)
                size = next((int(r.get("size", 0)) for r in tags.get("models", []) if r.get("name") == model), 0)
                if not size:
                    raise PipelineError("RESOURCE_BLOCKED", "无法取得所选非驻留润色模型的载入预算", blocked=True)
                estimate += int(size * 1.5)
        policy = model_policy(estimate, int(setting("client_max_rss_bytes", GIB // 2)), timeout)
        from .resources import sample_resources
        policy.preflight(sample_resources())
        folder = Path(scratch)
        request, output = folder / "request.json", folder / "response.json"
        write_json(request, {"endpoint": endpoint, "path": path, "body": body, "timeout": timeout, "output": str(output)})
        store = JobStore()
        job_id = os.environ.get("NEXTECHO_JOB_ID") or f"adhoc:{os.getpid()}:{hashlib.sha256(process_birth(os.getpid()).encode()).hexdigest()[:20]}"
        store.provider_start(endpoint, job_id)
        completed = False
        resources = None
        try:
            log = private_dir(store.root / "jobs" / job_id.replace(":", "_")) / f"provider-{folder.name}.log"
            resources = run_owned([sys.executable, str(Path(__file__).with_name("provider_worker.py")), str(request)],
                    log_path=log, policy=policy, cancel=current_cancelled,
                    interval=float(setting("resource_sample_seconds", 2)))
            if not output.is_file() or output.stat().st_size > 32 * 1024**2:
                raise PipelineError("PROTOCOL_ERROR", "本地模型客户端结果缺失或过大", stage="polish")
            result = json.loads(output.read_text())
            completed = bool(result.get("completed"))
            if result.get("error_code"):
                message = f"本地模型服务返回HTTP {result['http_status']}" if result.get("http_status") else "本地模型服务未完成有效请求"
                raise PipelineError(result["error_code"], message, stage="polish", blocked=True)
            try:
                return json.loads(result["payload"])
            except (ValueError, KeyError) as exc:
                raise PipelineError("INVALID_RESPONSE", "本地模型服务返回无效JSON", stage="polish") from exc
        finally:
            store.provider_end(endpoint, completed=completed)
            if CONTEXT.get() and resources:
                CONTEXT.get().resources.append({"provider": endpoint, "model": model,
                        "client": resources, "external_model_rss_bytes": None,
                        "host_monitor_log": str(log.with_suffix(".resources.json"))})


def ollama_request(path: str, payload: dict, *, endpoint: str, timeout: float):
    keep_alive = payload.get("keep_alive")
    if path in {"/api/chat", "/api/generate"} and (
        isinstance(keep_alive, (int, float)) and keep_alive == 0 or
        isinstance(keep_alive, str) and re.fullmatch(r"\s*0+(?:\.0+)?(?:ns|us|µs|ms|s|m|h)?\s*", keep_alive)
    ):
        raise PipelineError("OWNERSHIP_REQUIRED", "借用模型请求不能携带卸载指令", blocked=True)
    if path == "/api/chat":
        return generate_request(endpoint, path, payload, timeout=min(timeout, float(setting("request_timeout_seconds", timeout))))
    return request_json(endpoint, path, payload, timeout=timeout)


def local_chat_json(*, endpoint, model, system, user, schema, temperature=0):
    mode = setting("structured_mode", "prompt-json-validated")
    max_tokens = int(setting("max_output_tokens", 4096))
    context_tokens = int(setting("context_tokens", 16384))
    if max_tokens <= 0 or context_tokens <= max_tokens:
        raise ValueError("Invalid local model token budget")
    system = system + "\n返回一个纯JSON对象；不要Markdown代码围栏、解释或思考过程。符合此Schema：" + json.dumps(schema, ensure_ascii=False)
    if len((system + user).encode()) + max_tokens > context_tokens:
        raise ValueError("Input exceeds conservative model context budget")
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature, "max_tokens": max_tokens, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    if mode == "native-schema":
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "nextecho", "schema": schema, "strict": True}}
    payload = generate_request(endpoint, "/v1/chat/completions", body, timeout=float(setting("request_timeout_seconds", 120)))
    context = CONTEXT.get()
    usage_row = None
    if context:
        usage = payload.get("usage") or {}
        usage_row = {"model": model, "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")}
        context.usage.append(usage_row)
    choices = payload.get("choices") or []
    if not choices or choices[0].get("finish_reason") == "length":
        raise ValueError("Local model returned no complete content")
    # Never treat reasoning_content as transcript output.
    content = str((choices[0].get("message") or {}).get("content") or "")
    parsed, envelope = decode_model_json(content)
    validate_json_schema(parsed, schema)
    if usage_row is not None:
        usage_row["response_envelope"] = envelope
    return parsed
