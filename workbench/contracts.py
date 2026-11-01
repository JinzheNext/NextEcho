"""Versioned requests, immutable execution plans, and public failures."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from .config import NextEchoConfig
from .model_profiles import PROFILES


class PipelineError(RuntimeError):
    def __init__(self, code: str, message: str, *, stage: str = "prepare", blocked: bool = False):
        super().__init__(message)
        self.code, self.stage, self.blocked = code, stage, blocked


@dataclass(frozen=True)
class PipelineRequest:
    sources: tuple[str, ...]
    profile: str = "balanced"
    postprocess: str = "polish"
    speaker_mode: str = "off"
    timestamps: str = "segment"
    language: str = "zh"
    max_seconds: int = 0
    limit: int = 1
    asr_model: str = ""
    asr_engine: str = ""
    polish_model: str = ""
    allow_fallback: bool = False

    def validate(self) -> None:
        if not self.sources or len(self.sources) > 20 or any(not s.strip() for s in self.sources):
            raise PipelineError("INVALID_INPUT", "需要1至20个媒体输入")
        for key, allowed in {
            "profile": {"fast", "balanced", "quality"},
            "postprocess": {"off", "polish", "polish-and-summary"},
            "speaker_mode": {"off", "auto", "required"},
            "timestamps": {"segment", "precise"},
            "asr_engine": {"", "fun-asr-nano", "sensevoice", "qwen3-asr", "whisper-cpp"},
        }.items():
            if getattr(self, key) not in allowed:
                raise PipelineError("INVALID_INPUT", f"不支持的 {key}")
        if self.max_seconds < 0 or not 1 <= self.limit <= 20:
            raise PipelineError("INVALID_INPUT", "max-seconds不能为负数，limit必须在1至20内")


@dataclass(frozen=True)
class PipelinePlan:
    request: PipelineRequest
    asr_profile: str
    polish_profile: str
    effective: dict[str, Any]
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def plan_request(request: PipelineRequest, config: NextEchoConfig) -> PipelinePlan:
    request.validate()
    defaults = {
        "fast": ("sensevoice-int8", "qwen35-2b"),
        "balanced": ("fun-asr-nano-pt", "qwen35-4b"),
        "quality": ("fun-asr-nano-pt", "qwen38-27b-splash"),
    }
    asr, polish = defaults[request.profile]
    asr = config.get_path("profiles", request.profile, "asr", default=asr)
    if request.asr_engine in {"sensevoice", "fun-asr-nano"}:
        asr = "sensevoice-int8" if request.asr_engine == "sensevoice" else "fun-asr-nano-pt"
    polish = request.polish_model or config.get_path("profiles", request.profile, "polish", default=polish)
    if asr not in PROFILES or polish not in PROFILES or PROFILES[asr].task != "asr" or PROFILES[polish].task != "polish":
        raise PipelineError("INVALID_INPUT", "未知的模型档案，请配置已登记的档案ID")
    effective = config.to_dict()
    engine = request.asr_engine or PROFILES[asr].engine
    model = request.asr_model or PROFILES[asr].model
    if engine == "whisper-cpp":
        model = request.asr_model or config.get_path("asr", "whisper_fallback", default="large-v3-turbo-q5_0")
    elif engine == "qwen3-asr":
        model = request.asr_model or config.get_path("asr", "accurate_model", default="Qwen/Qwen3-ASR-1.7B")
    effective["resolved"] = {
        "asr_engine": engine,
        "asr_model": model,
        "polish_model": PROFILES[polish].model,
        "polish_provider": PROFILES[polish].engine,
    }
    return PipelinePlan(request, asr, polish, effective)
