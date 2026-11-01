"""Explicit text-layer adoption; physical files alone are not current outputs."""
from __future__ import annotations

from pathlib import Path

from .contracts import PipelineError
from .io_utils import file_hash

SPEAKER_PATHS = {"dialogue_json_path": "transcript.dialogue.json", "speaker_json_path": "transcript.speakers.json"}


def current_speaker_inputs(item: dict, *, enabled=True) -> dict:
    return {key: item[key] for key in SPEAKER_PATHS if item.get(key)} if enabled else {}


def select_text_input(root: Path, artifacts: dict | None = None) -> dict:
    """None: historical explicit enhance; {}: raw-only; dict: current stage paths."""
    if artifacts is not None and (not isinstance(artifacts, dict) or any(k not in SPEAKER_PATHS for k in artifacts)):
        raise PipelineError("INVALID_INPUT", "润色输入选择包含未知产物")
    if artifacts is None:
        artifacts = {key: str(root / name) for key, name in SPEAKER_PATHS.items() if (root / name).is_file()}
        policy = "legacy-validated-discovery"
    else:
        policy = "explicit-current-stage" if artifacts else "raw-only"
    paths = []
    for key, raw in artifacts.items():
        path = Path(raw)
        expected = (root / SPEAKER_PATHS[key]).resolve()
        if not path.is_file() or path.resolve() != expected or expected.parent != root.resolve():
            raise PipelineError("CACHE_MISMATCH", "本轮润色输入路径无效、缺失或越出当前录音", blocked=True)
        paths.append(path)
    dialogue = "dialogue_json_path" in artifacts
    primary = "transcript.dialogue.json" if dialogue else "transcript.speakers.json" if artifacts else "transcript.json"
    if dialogue:
        paths += [root / name for name in ("dialogue.integrity.json", "speaker.identity-map.json") if (root / name).is_file()]
    paths += [root / name for name in ("transcript.json", "transcript.txt", "source.context.json") if (root / name).is_file()]
    return {"schema_version": 1, "policy": policy, "layer": "acoustic-dialogue" if dialogue else "speaker-artifacts" if artifacts else "raw-asr",
            "primary_name": primary, "artifact_hashes": {p.name: file_hash(p) for p in paths}}
