"""Local speaker enrollment and profile storage.

Enrollment is deliberately separate from diarization and identity mapping.  A
profile contains a display name, consent/provenance and a pointer to a local
embedding artifact; it never contains a copy of the source audio, an access
token, or the embedding vector itself in JSON.  Embedding files and the profile
index are created with owner-only permissions by default.

The only fully implemented embedding backend in this repository is the
optional Resemblyzer backend.  The backend name is configuration-driven so a
future ONNX/3D-Speaker implementation can be added without changing the
profile file contract.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import sys
import tempfile
import unicodedata
import wave
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

from .config import NextEchoConfig, load_config, user_config_path


SCHEMA_VERSION = 1
DEFAULT_THRESHOLD = 0.78
DEFAULT_EMBEDDING_BACKEND = "resemblyzer"
DEFAULT_EMBEDDING_MODEL = "resemblyzer/VoiceEncoder"
PROFILE_ENV_VARS = ("NEXTECHO_SPEAKER_PROFILES", "NEXTECHO_SPEAKER_PROFILES_PATH")


class SpeakerProfileError(ValueError):
    """Raised when enrollment input or the local profile store is invalid."""


def profile_store_path(
    config: NextEchoConfig | None = None,
    *,
    profiles_path: str | os.PathLike[str] | None = None,
    env: dict[str, str] | None = None,
) -> Path:
    """Resolve the local profile index path without creating it."""

    if profiles_path is not None:
        candidate = Path(profiles_path).expanduser()
    else:
        environ = env if env is not None else os.environ
        configured_env = next((str(environ.get(name) or "").strip() for name in PROFILE_ENV_VARS if environ.get(name)), "")
        configured = configured_env or str((config or load_config()).get_path("speaker_identity", "profiles_path", default="") or "")
        if configured:
            candidate = Path(configured).expanduser()
        else:
            # Keep biometric data outside the repository and next to the
            # platform-specific user config directory.
            base = (config.user_path.parent if config and config.user_path else user_config_path().parent)
            candidate = base / "speaker_profiles.json"
    if not candidate.is_absolute():
        base = (config.user_path.parent if config and config.user_path else user_config_path().parent)
        candidate = base / candidate
    return candidate.resolve()


def embedding_store_path(profile_path: Path, person_id: str) -> Path:
    """Return an embedding path constrained to a private sibling directory."""

    safe_id = _safe_person_id(person_id)
    return profile_path.parent / "speaker_embeddings" / f"{safe_id}.npy"


def load_profiles(
    config: NextEchoConfig | None = None,
    *,
    profiles_path: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Read profile metadata; missing stores are treated as an empty list."""

    path = profile_store_path(config, profiles_path=profiles_path)
    if not path.exists():
        return []
    if not path.is_file():
        raise SpeakerProfileError(f"Speaker profile path is not a file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpeakerProfileError(f"Cannot read speaker profiles: {path}: {exc}") from exc
    if isinstance(payload, list):
        # Read the earliest ad-hoc shape for a painless upgrade, but all new
        # writes use the versioned object below.
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("profiles") or []
    else:
        raise SpeakerProfileError(f"Invalid speaker profile index: {path}")
    if not isinstance(rows, list):
        raise SpeakerProfileError(f"Invalid speaker profile list: {path}")
    return [dict(item) for item in rows if isinstance(item, dict)]


def list_profiles(
    config: NextEchoConfig | None = None,
    *,
    profiles_path: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Return metadata only, never embedding vectors or audio bytes."""

    return load_profiles(config, profiles_path=profiles_path)


def enroll_speaker(
    display_name: str,
    sample_paths: Sequence[str | os.PathLike[str]],
    *,
    consent: bool = False,
    backend: str = "auto",
    config: NextEchoConfig | None = None,
    profiles_path: str | os.PathLike[str] | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Generate and persist one local voice profile.

    ``consent`` is intentionally required to be true.  A caller can enroll
    several samples at once; their normalized embeddings are averaged and
    normalized again for matching.  Existing profiles with the same person ID
    are replaced while retaining their original creation timestamp.
    """

    name = str(display_name or "").strip()
    if not name:
        raise SpeakerProfileError("display_name is required")
    if not consent:
        raise SpeakerProfileError("explicit consent is required before storing a speaker embedding")
    paths = _validate_sample_paths(sample_paths)
    config_obj = config or load_config()
    selected_backend = resolve_embedding_backend(backend, config_obj)
    if selected_backend != "resemblyzer":
        raise SpeakerProfileError(f"speaker embedding backend is not implemented: {selected_backend}")

    embedding, model_name = _embed_with_resemblyzer(paths, config_obj)
    person_id = _person_id_for_name(name)
    store_path = profile_store_path(config_obj, profiles_path=profiles_path)
    embedding_path = embedding_store_path(store_path, person_id)
    embedding_digest = _write_embedding(embedding_path, embedding)
    sample_digests = [_sha256_file(path) for path in paths]
    now = _now_iso()
    profiles = load_profiles(config_obj, profiles_path=store_path)
    existing = next((item for item in profiles if str(item.get("person_id") or "") == person_id), None)
    created_at = str(existing.get("created_at") or now) if existing else now
    configured_threshold = threshold
    if configured_threshold is None:
        configured_threshold = float(config_obj.get_path("speaker_identity", "profile_threshold", default=DEFAULT_THRESHOLD) or DEFAULT_THRESHOLD)
    configured_threshold = max(0.0, min(1.0, float(configured_threshold)))
    profile: dict[str, Any] = {
        "person_id": person_id,
        "display_name": name,
        # Paths are provenance only; source bytes are never copied into the
        # profile directory or JSON index.
        "sample_paths": [str(path) for path in paths],
        "sample_count": len(paths),
        "sample_digests": sample_digests,
        "embedding_backend": selected_backend,
        "embedding_model": model_name,
        "embedding_path": str(embedding_path),
        "embedding_digest": embedding_digest,
        "threshold": configured_threshold,
        "consent": True,
        "created_at": created_at,
        "updated_at": now,
    }
    profiles = [item for item in profiles if str(item.get("person_id") or "") != person_id]
    profiles.append(profile)
    profiles.sort(key=lambda item: (str(item.get("display_name") or ""), str(item.get("person_id") or "")))
    _write_profile_index(store_path, profiles)
    return dict(profile)


def remove_profile(
    person_id: str,
    *,
    config: NextEchoConfig | None = None,
    profiles_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Remove profile metadata and its local embedding artifact."""

    clean_id = str(person_id or "").strip()
    if not clean_id:
        raise SpeakerProfileError("person_id is required")
    config_obj = config or load_config()
    store_path = profile_store_path(config_obj, profiles_path=profiles_path)
    profiles = load_profiles(config_obj, profiles_path=store_path)
    removed = next((item for item in profiles if str(item.get("person_id") or "") == clean_id), None)
    if removed is None:
        raise FileNotFoundError(f"speaker profile not found: {clean_id}")
    remaining = [item for item in profiles if str(item.get("person_id") or "") != clean_id]
    _write_profile_index(store_path, remaining)
    embedding_value = removed.get("embedding_path")
    if embedding_value:
        embedding_path = Path(str(embedding_value)).expanduser()
        # Only delete a path under the profile store's embedding directory;
        # hand-edited metadata cannot make ``remove`` delete arbitrary files.
        expected_dir = (store_path.parent / "speaker_embeddings").resolve()
        try:
            is_safe = embedding_path.resolve().parent == expected_dir
        except OSError:
            is_safe = False
        if is_safe:
            embedding_path.unlink(missing_ok=True)
    return {"status": "removed", "person_id": clean_id, "display_name": removed.get("display_name", "")}


def resolve_embedding_backend(backend: str = "auto", config: NextEchoConfig | None = None) -> str:
    config_obj = config or load_config()
    requested = str(backend or "auto").strip().lower().replace("_", "-")
    if requested == "auto":
        requested = str(config_obj.get_path("speaker_identity", "embedding_backend", default=DEFAULT_EMBEDDING_BACKEND) or DEFAULT_EMBEDDING_BACKEND).strip().lower().replace("_", "-")
    aliases = {"resemblyzer": "resemblyzer", "speaker-embedding": "resemblyzer", "voiceencoder": "resemblyzer"}
    selected = aliases.get(requested, requested)
    if selected == "resemblyzer" and not _package_available("resemblyzer"):
        raise SpeakerProfileError("Resemblyzer is not installed; install requirements-speakers-embedding.txt or choose another backend")
    return selected


def load_profile_embedding(
    profile: dict[str, Any],
    *,
    profiles_path: str | os.PathLike[str] | None = None,
    config: NextEchoConfig | None = None,
) -> Any:
    """Load a local vector for a future identity matcher after verification."""

    embedding_value = str(profile.get("embedding_path") or "")
    if not embedding_value:
        raise SpeakerProfileError("profile has no embedding_path")
    path = Path(embedding_value).expanduser().resolve()
    store = profile_store_path(config, profiles_path=profiles_path)
    expected_dir = (store.parent / "speaker_embeddings").resolve()
    if path.parent != expected_dir:
        raise SpeakerProfileError("embedding path is outside the local profile store")
    if not path.is_file():
        raise FileNotFoundError(path)
    if profile.get("embedding_digest") and _sha256_file(path) != str(profile["embedding_digest"]):
        raise SpeakerProfileError("embedding digest mismatch")
    try:
        import numpy as np

        return np.load(path, allow_pickle=False)
    except ImportError as exc:
        raise SpeakerProfileError("numpy is required to load a speaker embedding") from exc


def _embed_with_resemblyzer(paths: list[Path], config: NextEchoConfig) -> tuple[Any, str]:
    try:
        module = importlib.import_module("resemblyzer")
    except ImportError as exc:
        raise SpeakerProfileError("Resemblyzer is not installed; install requirements-speakers-embedding.txt") from exc
    try:
        import numpy as np
    except ImportError as exc:
        raise SpeakerProfileError("numpy is required for speaker enrollment") from exc
    encoder_cls = getattr(module, "VoiceEncoder", None)
    if encoder_cls is None:
        raise SpeakerProfileError("installed Resemblyzer does not expose VoiceEncoder")
    try:
        encoder = encoder_cls("cpu")
    except TypeError:
        encoder = encoder_cls()
    preprocess = getattr(module, "preprocess_wav", None)
    vectors: list[Any] = []
    for path in paths:
        if preprocess is not None:
            try:
                waveform = preprocess(str(path))
            except TypeError:
                waveform = preprocess(path)
        else:
            waveform = _read_wav_16k(path, np)
        try:
            vector = encoder.embed_utterance(waveform, return_partials=False)
        except TypeError:
            vector = encoder.embed_utterance(waveform)
        values = np.asarray(vector, dtype=np.float32).reshape(-1)
        if values.size == 0 or not np.isfinite(values).all():
            raise SpeakerProfileError(f"Resemblyzer returned an invalid embedding for {path}")
        norm = float(np.linalg.norm(values))
        if norm <= 0:
            raise SpeakerProfileError(f"Resemblyzer returned a zero embedding for {path}")
        vectors.append(values / norm)
    average = np.mean(np.vstack(vectors), axis=0)
    norm = float(np.linalg.norm(average))
    if norm <= 0 or not np.isfinite(average).all():
        raise SpeakerProfileError("could not construct a non-zero speaker embedding")
    model_name = str(config.get_path("speaker_identity", "embedding_model", default=DEFAULT_EMBEDDING_MODEL) or DEFAULT_EMBEDDING_MODEL)
    return (average / norm).astype(np.float32), model_name


def _read_wav_16k(path: Path, np: Any) -> Any:
    try:
        with wave.open(str(path), "rb") as handle:
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            raw = handle.readframes(handle.getnframes())
    except (OSError, wave.Error) as exc:
        raise SpeakerProfileError(f"Resemblyzer preprocess_wav is unavailable and sample is not readable WAV: {path}") from exc
    if sample_width != 2:
        raise SpeakerProfileError(f"only 16-bit PCM WAV can be enrolled without Resemblyzer preprocess_wav: {path}")
    values = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        values = values.reshape(-1, channels).mean(axis=1)
    if sample_rate != 16000 and values.size:
        target_size = max(1, int(round(values.size * 16000 / sample_rate)))
        source_x = np.linspace(0.0, 1.0, values.size, endpoint=False)
        target_x = np.linspace(0.0, 1.0, target_size, endpoint=False)
        values = np.interp(target_x, source_x, values).astype(np.float32)
    return values


def _validate_sample_paths(sample_paths: Sequence[str | os.PathLike[str]]) -> list[Path]:
    if isinstance(sample_paths, (str, bytes, os.PathLike)):
        sample_paths = [sample_paths]  # type: ignore[list-item]
    paths: list[Path] = []
    for value in sample_paths or []:
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size <= 0:
            raise SpeakerProfileError(f"enrollment sample is empty: {path}")
        if path in paths:
            continue
        paths.append(path)
    if not paths:
        raise SpeakerProfileError("at least one enrollment sample is required")
    return paths


def _package_available(name: str) -> bool:
    if name in sys.modules and sys.modules[name] is not None:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        # Test doubles and namespace packages may be present in sys.modules
        # without a ModuleSpec.
        return name in sys.modules


def _safe_person_id(person_id: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(person_id or "")).strip()
    pieces: list[str] = []
    for char in normalized:
        if char.isalnum() or char in {"-", "_"}:
            pieces.append(char)
        elif not pieces or pieces[-1] != "_":
            pieces.append("_")
    value = "".join(pieces).strip("._-")[:80]
    return value or hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _person_id_for_name(name: str) -> str:
    return f"person_{_safe_person_id(name).lower()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_embedding(path: Path, embedding: Any) -> str:
    try:
        import numpy as np
    except ImportError as exc:
        raise SpeakerProfileError("numpy is required to persist a speaker embedding") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    _chmod_private(path.parent)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".tmp", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, np.asarray(embedding, dtype=np.float32), allow_pickle=False)
        _chmod_private(temp_path)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)
    _chmod_private(path)
    return _sha256_file(path)


def _write_profile_index(path: Path, profiles: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _chmod_private(path.parent)
    payload = {"schema_version": SCHEMA_VERSION, "profiles": profiles}
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=str(path.parent))
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        _chmod_private(temp_path)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)
    _chmod_private(path)


def _chmod_private(path: Path) -> None:
    try:
        mode = 0o700 if path.is_dir() else 0o600
        path.chmod(mode)
    except OSError:
        # Windows may not expose POSIX mode bits; the file remains local and
        # the write itself is still atomic.
        return


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


# Small compatibility aliases keep the module easy to call from scripts and
# match the dynamic CLI entry-point names.
enroll_profile = enroll_speaker
get_profiles = list_profiles
delete_profile = remove_profile


__all__ = [
    "DEFAULT_EMBEDDING_BACKEND",
    "DEFAULT_EMBEDDING_MODEL",
    "SCHEMA_VERSION",
    "SpeakerProfileError",
    "embedding_store_path",
    "delete_profile",
    "enroll_profile",
    "enroll_speaker",
    "get_profiles",
    "list_profiles",
    "load_profile_embedding",
    "load_profiles",
    "profile_store_path",
    "remove_profile",
    "resolve_embedding_backend",
]
