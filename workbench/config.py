"""Central, local-first NextEcho configuration.

The repository ships a small TOML file with safe defaults.  A user override is
read from an OS-specific configuration directory, or from ``NEXTECHO_CONFIG``
/``--config`` when explicitly selected.  This module deliberately does not
import the ``toml`` package and has a small standards-oriented fallback parser
for Python versions without ``tomllib`` (or ``tomli``).

Configuration is data only: no expressions, imports, environment expansion, or
commands are evaluated while parsing it.
"""

from __future__ import annotations

import copy
import json
import os
import platform
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config" / "nextecho.toml"
CONFIG_ENV_VARS = ("NEXTECHO_CONFIG", "NEXTECHO_CONFIG_PATH")
MAX_CONFIG_BYTES = 1 * 1024 * 1024


DEFAULTS: dict[str, Any] = {
    "schema_version": 1,
    "hardware": {
        "profile": "16gb-apple-silicon",
        "profile_source": "",
        "reserve_memory_gb": 3.0,
        "max_swap_delta_gb": 1.0,
        "memory_pressure_max": "warn",
    },
    "asr": {
        "engine": "auto",
        "language": "zh",
        "accurate_model": "aufklarer/Qwen3-ASR-1.7B-MLX-5bit",
        "fast_model": "Qwen/Qwen3-ASR-0.6B",
        "forced_aligner": "Qwen/Qwen3-ForcedAligner-0.6B",
        "whisper_fallback": "large-v3-turbo-q5_0",
        "experimental_accurate_model": True,
    },
    "diarization": {
        "backend": "auto",
        "default_mode": "auto",
        "benchmark_candidates": ["pyannote-community-1", "sherpa-onnx", "resemblyzer"],
        "fallback_order": ["pyannote-community-1", "sherpa-onnx", "resemblyzer"],
        "min_speakers": 2,
        "max_speakers": 6,
        "allow_unknown": True,
        "use_exclusive_timeline": True,
        "worker_python": "3.12",
    },
    "speaker_identity": {
        "enable_metadata_hints": True,
        "enable_self_introduction_hints": True,
        "enable_voice_enrollment": False,
        "embedding_backend": "resemblyzer",
        "embedding_model": "resemblyzer/VoiceEncoder",
        "profiles_path": "",
        "profile_threshold": 0.78,
        "minimum_confidence": 0.90,
        "text_anchor_scope": "cluster_mapping_only",
    },
    "postprocess": {
        "profile": "fast",
        "provider": "selection",
        "selection_source": "",
        "preferred_model_id": "gemma-e4b-iq4xs",
        "fallback_model": "qwen3:4b",
        "context_tokens": 16384,
        "polish_target_chars": 1800,
        "polish_max_chars": 4000,
        "keep_alive_seconds": 600,
    },
    "context": {
        "shownotes": True,
        "comments_top_n": 20,
        "enforce_transcript_coverage": True,
        "comments_are_evidence": False,
    },
    "runtime": {
        "heavy_task_concurrency": 1,
        "heartbeat_seconds": 15,
        "stage_checkpointing": True,
        "release_model_after_stage": True,
    },
}


class ConfigError(ValueError):
    """Raised when a configuration file is malformed or too large."""


class NextEchoConfig(Mapping[str, Any]):
    """Read-only mapping plus provenance for one merged configuration.

    Keeping mapping semantics makes the object convenient for existing callers
    (``config["asr"]["engine"]``) while the attributes make source reporting
    explicit for CLI and doctor output.
    """

    def __init__(
        self,
        data: Mapping[str, Any],
        *,
        source_paths: list[str] | None = None,
        source_kinds: list[str] | None = None,
        repo_path: Path = DEFAULT_CONFIG_PATH,
        user_path: Path | None = None,
    ) -> None:
        self._data = copy.deepcopy(dict(data))
        self.source_paths = list(source_paths or [])
        self.source_kinds = list(source_kinds or [])
        self.repo_path = Path(repo_path)
        self.user_path = Path(user_path) if user_path else None

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"NextEchoConfig(source={self.source!r}, keys={list(self._data)!r})"

    @property
    def data(self) -> dict[str, Any]:
        """Return a defensive plain-dict copy for serialization."""

        return copy.deepcopy(self._data)

    @property
    def source(self) -> str:
        """Human-readable source label for the effective configuration."""

        if not self.source_paths:
            return "built-in defaults"
        if len(self.source_paths) == 1 and self.source_kinds[:1] == ["repo-default"]:
            return f"repo-default:{self.source_paths[0]}"
        return "+".join(
            f"{kind}:{path}" for kind, path in zip(self.source_kinds, self.source_paths)
        )

    @property
    def source_label(self) -> str:
        """Short source name suitable for JSON/UI labels."""

        if "user-override" in self.source_kinds:
            return "user-override"
        if "repo-default" in self.source_kinds:
            return "repo-default"
        return "built-in-defaults"

    def to_dict(self) -> dict[str, Any]:
        return self.data

    def get_path(self, *keys: str, default: Any = None) -> Any:
        value: Any = self._data
        for key in keys:
            if not isinstance(value, Mapping) or key not in value:
                return default
            value = value[key]
        return value


def default_config_path() -> Path:
    return DEFAULT_CONFIG_PATH


def user_config_path(*, system: str | None = None, env: Mapping[str, str] | None = None) -> Path:
    """Return the OS-specific user override path without creating anything."""

    environ = env if env is not None else os.environ
    system_name = (system or platform.system()).lower()
    if system_name == "darwin":
        return Path.home() / "Library" / "Application Support" / "NextEcho" / "config.toml"
    if system_name == "windows":
        base = environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "NextEcho" / "config.toml"
    xdg = environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "nextecho" / "config.toml"


def _selected_override(
    config_path: str | os.PathLike[str] | None,
    *,
    env: Mapping[str, str],
    system: str | None = None,
) -> tuple[Path | None, str]:
    if config_path is not None:
        return Path(config_path).expanduser(), "explicit"
    for name in CONFIG_ENV_VARS:
        value = str(env.get(name) or "").strip()
        if value:
            return Path(value).expanduser(), "environment"
    return user_config_path(system=system, env=env), "user-override"


def load_config(
    config_path: str | os.PathLike[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    repo_path: str | os.PathLike[str] | None = None,
    user_path: str | os.PathLike[str] | None = None,
    strict: bool = False,
) -> NextEchoConfig:
    """Load repository defaults and merge one external user override.

    Missing files are normal: the repository default may be absent in an
    installed package and the user override is optional.  ``strict=True`` is
    useful for CLI validation and raises for an explicitly selected path that
    cannot be read; the normal mode records only files that actually exist.
    """

    environ = env if env is not None else os.environ
    selected_repo = Path(repo_path).expanduser() if repo_path is not None else DEFAULT_CONFIG_PATH
    if user_path is not None:
        selected_user = Path(user_path).expanduser()
        override_kind = "user-override"
    else:
        selected_user, override_kind = _selected_override(config_path, env=environ)

    merged: dict[str, Any] = copy.deepcopy(DEFAULTS)
    source_paths: list[str] = []
    source_kinds: list[str] = []

    if selected_repo.exists():
        repo_data = _read_config_file(selected_repo)
        _deep_merge(merged, repo_data)
        source_paths.append(str(selected_repo.resolve()))
        source_kinds.append("repo-default")
    elif strict and repo_path is not None:
        raise ConfigError(f"Configuration file does not exist: {selected_repo}")

    # An explicit path may intentionally point at the repository file.  Avoid
    # parsing it twice and avoid reporting a misleading override source.
    if selected_user is not None:
        same_as_repo = False
        try:
            same_as_repo = selected_user.resolve() == selected_repo.resolve()
        except OSError:
            pass
        if not same_as_repo and selected_user.exists():
            override_data = _read_config_file(selected_user)
            _deep_merge(merged, override_data)
            source_paths.append(str(selected_user.resolve()))
            source_kinds.append("explicit" if override_kind == "explicit" else "user-override")
        elif strict and (config_path is not None or override_kind == "environment"):
            raise ConfigError(f"Configuration file does not exist: {selected_user}")

    return NextEchoConfig(
        merged,
        source_paths=source_paths,
        source_kinds=source_kinds,
        repo_path=selected_repo,
        user_path=selected_user,
    )


def load_config_dict(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility helper for callers that only need plain data."""

    return load_config(*args, **kwargs).to_dict()


def _read_config_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Configuration path is not a regular file: {path}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ConfigError(f"Cannot stat configuration file {path}: {exc}") from exc
    if size > MAX_CONFIG_BYTES:
        raise ConfigError(f"Configuration file is too large (>{MAX_CONFIG_BYTES} bytes): {path}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"Cannot read configuration file {path}: {exc}") from exc
    try:
        return _parse_toml(raw.decode("utf-8"), source=str(path))
    except UnicodeDecodeError as exc:
        raise ConfigError(f"Configuration file must be UTF-8: {path}") from exc


def _parse_toml(text: str, *, source: str = "<config>") -> dict[str, Any]:
    """Parse TOML using stdlib, tomli, or a deliberately limited safe parser."""

    try:
        import tomllib  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        tomllib = None
    if tomllib is not None:
        try:
            return dict(tomllib.loads(text))
        except Exception as exc:
            raise ConfigError(f"Invalid TOML in {source}: {exc}") from exc
    try:
        import tomli  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return _parse_minimal_toml(text, source=source)
    try:
        return dict(tomli.loads(text))
    except Exception as exc:
        raise ConfigError(f"Invalid TOML in {source}: {exc}") from exc


def _parse_minimal_toml(text: str, *, source: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    table: list[str] = []
    pending = ""
    pending_line = 0
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if pending:
            pending += "\n" + raw_line
            if _balanced_value(pending):
                line = pending
                pending = ""
            else:
                continue
        else:
            line = raw_line
            if not _balanced_value(_strip_inline_comment(line)):
                pending = line
                pending_line = line_no
                continue
        line = _strip_inline_comment(line).strip()
        if not line:
            continue
        if line.startswith("[[") and line.endswith("]]"):
            raise ConfigError(f"Array-of-tables is unsupported in fallback parser ({source}:{line_no})")
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1].strip()
            table = _split_dotted_key(name, source=source, line_no=line_no)
            _ensure_table(result, table)
            continue
        if "=" not in line:
            raise ConfigError(f"Expected key = value ({source}:{line_no})")
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ConfigError(f"Empty key ({source}:{line_no})")
        path = table + _split_dotted_key(key, source=source, line_no=line_no)
        value = _parse_minimal_value(raw_value.strip(), source=source, line_no=line_no)
        _set_unique(result, path, value, source=source, line_no=line_no)
    if pending:
        raise ConfigError(f"Unterminated array or inline value ({source}:{pending_line})")
    return result


def _balanced_value(value: str) -> bool:
    quote = ""
    escaped = False
    depth = 0
    for char in value:
        if quote:
            if quote == '"' and escaped:
                escaped = False
            elif quote == '"' and char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if char in {'"', "'"}:
            quote = char
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
    return not quote and depth <= 0


def _strip_inline_comment(value: str) -> str:
    quote = ""
    escaped = False
    depth = 0
    for index, char in enumerate(value):
        if quote:
            if quote == '"' and escaped:
                escaped = False
            elif quote == '"' and char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
        elif char in {'"', "'"}:
            quote = char
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
        elif char == "#" and depth == 0:
            return value[:index]
    return value


def _split_dotted_key(value: str, *, source: str, line_no: int) -> list[str]:
    parts = [item.strip() for item in value.split(".")]
    if not parts or any(not item or any(char in item for char in "[]{}=\n") for item in parts):
        raise ConfigError(f"Invalid dotted key ({source}:{line_no})")
    cleaned: list[str] = []
    for item in parts:
        if len(item) >= 2 and item[0] in {'"', "'"} and item[-1] == item[0]:
            item = _parse_minimal_value(item, source=source, line_no=line_no)
        if not isinstance(item, str) or not item:
            raise ConfigError(f"Invalid key ({source}:{line_no})")
        cleaned.append(item)
    return cleaned


def _parse_minimal_value(value: str, *, source: str, line_no: int) -> Any:
    value = _strip_inline_comment(value).strip()
    if not value:
        raise ConfigError(f"Empty value ({source}:{line_no})")
    if value in {"true", "false"}:
        return value == "true"
    if value.startswith('"') and value.endswith('"'):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Invalid string ({source}:{line_no})") from exc
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        body = value[1:-1]
        return [_parse_minimal_value(item, source=source, line_no=line_no) for item in _split_top_level(body, ",") if item.strip()]
    if value.startswith("{") and value.endswith("}"):
        body = value[1:-1]
        inline: dict[str, Any] = {}
        for item in _split_top_level(body, ","):
            if "=" not in item:
                raise ConfigError(f"Invalid inline table ({source}:{line_no})")
            key, raw = item.split("=", 1)
            inline_key = _split_dotted_key(key.strip(), source=source, line_no=line_no)
            _set_unique(inline, inline_key, _parse_minimal_value(raw, source=source, line_no=line_no), source=source, line_no=line_no)
        return inline
    if re.fullmatch(r"[+-]?\d+", value):
        try:
            return int(value)
        except ValueError:
            pass
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][+-]?\d+)?", value) or re.fullmatch(r"[+-]?\d+[eE][+-]?\d+", value):
        try:
            return float(value)
        except ValueError:
            pass
    # TOML dates are not used by the shipped config.  Preserve unknown scalar
    # values as strings rather than evaluating arbitrary syntax.
    if any(char in value for char in "()$;"):
        raise ConfigError(f"Unsupported or unsafe scalar ({source}:{line_no})")
    return value


def _split_top_level(value: str, delimiter: str) -> list[str]:
    parts: list[str] = []
    start = 0
    quote = ""
    escaped = False
    depth = 0
    for index, char in enumerate(value):
        if quote:
            if quote == '"' and escaped:
                escaped = False
            elif quote == '"' and char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
        elif char in {'"', "'"}:
            quote = char
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
        elif char == delimiter and depth == 0:
            parts.append(value[start:index])
            start = index + 1
    parts.append(value[start:])
    return parts


def _ensure_table(result: dict[str, Any], path: list[str]) -> dict[str, Any]:
    current = result
    for key in path:
        value = current.get(key)
        if value is None:
            value = {}
            current[key] = value
        if not isinstance(value, dict):
            raise ConfigError(f"Table conflicts with scalar: {'.'.join(path)}")
        current = value
    return current


def _set_unique(result: dict[str, Any], path: list[str], value: Any, *, source: str, line_no: int) -> None:
    current = _ensure_table(result, path[:-1])
    key = path[-1]
    if key in current:
        raise ConfigError(f"Duplicate key: {'.'.join(path)} ({source}:{line_no})")
    current[key] = value


def _deep_merge(target: dict[str, Any], override: Mapping[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
