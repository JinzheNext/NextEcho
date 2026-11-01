"""Explicit local model identities; a model family is not a size tier."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import sys
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelProfile:
    id: str
    task: str
    engine: str
    model: str
    tier: str
    quantization: str
    required_files: tuple[str, ...] = ()
    timing_precision: str = "none"
    release_status: str = "candidate"
    structured_mode: str = "native-schema"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PROFILES = {
    "fun-asr-nano-pt": ModelProfile("fun-asr-nano-pt", "asr", "fun-asr-nano", "FunAudioLLM/Fun-ASR-Nano-2512", "accurate", "fp32-cpu", ("config.yaml", "model.pt", "Qwen3-0.6B/config.json", "Qwen3-0.6B/tokenizer.json", "Qwen3-0.6B/tokenizer_config.json"), "vad-segment"),
    "sensevoice-int8": ModelProfile("sensevoice-int8", "asr", "sensevoice", "SenseVoiceSmall", "fast", "int8", ("model.int8.onnx", "tokens.txt"), "vad-segment"),
    # Tags can be rebound locally; /api/tags and /api/show establish actual
    # digest/quantization in the audit, rather than a research-era label.
    "qwen35-2b": ModelProfile("qwen35-2b", "polish", "ollama", "qwen3.5:2b-mlx", "fast", "runtime-reported", structured_mode="prompt-json-validated"),
    "qwen35-4b": ModelProfile("qwen35-4b", "polish", "ollama", "qwen3.5:4b-mlx", "balanced", "runtime-reported", structured_mode="prompt-json-validated"),
    "qwen35-9b": ModelProfile("qwen35-9b", "polish", "ollama", "qwen3.5:9b-mlx", "quality", "runtime-reported", structured_mode="prompt-json-validated"),
    "qwen38-27b-splash": ModelProfile("qwen38-27b-splash", "polish", "local-chat-compatible", "qwen3.8-27b-local", "quality", "mlx-affine-4bit", structured_mode="prompt-json-validated"),
}

# Legacy names remain supported without assigning an entire family to a tier.
_TEXT_TIERS = {
    **{p.model.lower(): p.tier for p in PROFILES.values() if p.task == "polish"},
    "qwen3:4b": "balanced", "gemma-e4b-iq4xs": "balanced",
    "qwen3.5:9b": "quality", "qwen3.5:9b-q4_k_m": "quality",
}


def text_model_tier(model: str) -> str:
    return _TEXT_TIERS.get(str(model).lower(), "custom")


def validate_model_files(root: Path, required: tuple[str, ...]) -> dict[str, Any]:
    """Resolve the exact directory subsequently passed to the worker."""
    root = root.expanduser().resolve()
    missing = [name for name in required if not (root / name).is_file() or not (root / name).stat().st_size]
    return {"path": str(root), "complete": not missing, "missing_files": missing}


def native_asr_status(config) -> list[dict]:
    models = config.get("models", {}) if config else {}
    rows = []
    for key in ("fun-asr-nano-pt", "sensevoice-int8"):
        p = PROFILES[key]
        settings = models.get(key, {})
        files = validate_model_files(Path(settings.get("path", ".")), p.required_files)
        calibrated = all(float(settings.get(k, 0)) > 0 for k in ("estimated_bytes", "max_rss_bytes", "timeout_seconds"))
        identity_valid = False
        if files["complete"] and calibrated and settings.get("release_status") == "ready" and settings.get("calibration_files") and settings.get("runtime_versions"):
            from .capabilities import NATIVE_REQUIREMENTS, probe_environment
            from .io_utils import file_hash
            actual = probe_environment(settings.get("worker_python") or sys.executable, NATIVE_REQUIREMENTS[key])
            vad = Path(config.get("asr", {}).get("vad_model", ""))
            identity_valid = (actual["available"] and all(file_hash(Path(files["path"]) / name) == settings["calibration_files"].get(name) for name in p.required_files)
                and all(actual["versions"].get(name) == settings["runtime_versions"].get(name) for name in NATIVE_REQUIREMENTS[key])
                and vad.is_file() and file_hash(vad) == settings.get("vad_sha256"))
        rows.append({**p.to_dict(), **files, "model": settings.get("path", p.model),
                     "release_status": settings.get("release_status", "candidate"),
                     "resource_calibrated": calibrated, "calibration_identity_valid": identity_valid,
                     "eligible": bool(identity_valid and settings.get("release_status") == "ready")})
    return rows


def selection_decision(requested, *, qwen_model, whisper_model, qwen_eligible, native=()):
    if requested not in {"auto", "qwen3-asr", "whisper-cpp", "fun-asr-nano", "sensevoice"}:
        raise ValueError(f"Unsupported ASR engine: {requested}")
    selected = requested
    if requested == "auto":
        selected = next((r["engine"] for r in native if r.get("eligible")), "qwen3-asr" if qwen_eligible else "whisper-cpp")
    model = qwen_model if selected == "qwen3-asr" else whisper_model
    if selected in {"fun-asr-nano", "sensevoice"}:
        model = next((r["model"] for r in native if r["engine"] == selected), "")
    return {"engine": selected, "model": model, "requested_engine": requested,
            "reason": "explicit" if requested != "auto" else "eligible-local-profile"}


def complete_hf_snapshot(model: str) -> Path | None:
    """Resolve one complete snapshot, including every indexed weight shard."""
    local = Path(model).expanduser()
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub" / ("models--" + model.replace("/", "--"))
    candidates = [local] if local.is_dir() else sorted((hub / "snapshots").glob("*"), reverse=True)
    ref = hub / "refs/main"
    if ref.is_file():
        preferred = hub / "snapshots" / ref.read_text().strip()
        candidates = [preferred, *[p for p in candidates if p != preferred]]
    for path in candidates:
        if not (path / "config.json").is_file():
            continue
        if not any((path / name).is_file() for name in ("tokenizer.json", "vocab.json", "vocab.txt", "tokens.txt", "tokenizer.model")):
            continue
        indices = list(path.glob("*.index.json"))
        try:
            weights = {str(v) for index in indices for v in json.loads(index.read_text()).get("weight_map", {}).values()}
        except (OSError, ValueError):
            continue
        if not weights:
            weights = {p.name for p in path.glob("*.safetensors")} | {p.name for p in path.glob("*.bin")}
            if any("-of-" in name for name in weights):
                continue
        if weights and all(not Path(name).is_absolute() and ".." not in Path(name).parts and
                           (path / name).is_file() and (path / name).resolve().suffix != ".incomplete" and
                           (path / name).stat().st_size > 0 for name in weights):
            return path.resolve()
    return None
