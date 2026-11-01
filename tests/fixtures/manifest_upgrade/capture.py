"""Regenerate the golden manifest-upgrade fixtures.

Run from the repository root with the baseline implementation intact::

    .venv312/bin/python tests/fixtures/manifest_upgrade/capture.py

Each case directory receives:

* ``manifest.json``   – the upgraded-run *input* manifest, paths replaced with
  the ``<RUN_DIR>`` placeholder so the test can materialize it anywhere.
* ``run_config.json`` – the run config consumed by the upgrade (when present).
* ``golden.json``     – the exact output of the baseline ``_upgrade_manifest_v3``
  with ``<RUN_DIR>`` placeholders and volatile resource fields stripped.

The golden files are the behavior lock for the R3 manifest extraction: the
replacement implementation must reproduce them field for field.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from app import _upgrade_manifest_v3  # noqa: E402  (baseline implementation)

FIXTURE_ROOT = Path(__file__).resolve().parent
PLACEHOLDER = "<RUN_DIR>"
VOLATILE_KEYS = {"captured_at", "peak_rss_bytes", "peak_rss_delta_bytes"}

REAL_RUNS = {
    "v1_real_20260613": REPO_ROOT / "outputs" / "transcriptions" / "run_20260613_141452_0d89dd83",
    "v2_real_20260910": REPO_ROOT / "outputs" / "transcriptions" / "final_xiaoyuzhou_5m_20260910",
}


def _rewrite_paths(value: object, *, source: str, target: str) -> object:
    """Rewrite run-dir rooted path strings between real and placeholder form."""

    if isinstance(value, dict):
        return {key: _rewrite_paths(child, source=source, target=target) for key, child in value.items()}
    if isinstance(value, list):
        return [_rewrite_paths(child, source=source, target=target) for child in value]
    if isinstance(value, str) and value.startswith(source):
        return target + value[len(source):]
    return value


def _strip_volatile(value: object) -> object:
    if isinstance(value, dict):
        return {key: _strip_volatile(child) for key, child in value.items() if key not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [_strip_volatile(child) for child in value]
    return value


def _normalize_output(manifest: dict, run_dir: Path) -> dict:
    for prefix in {str(run_dir), str(run_dir.resolve())}:
        manifest = _rewrite_paths(manifest, source=prefix, target=PLACEHOLDER)
    return _strip_volatile(manifest)


def _capture_case(name: str, manifest: dict, run_config: dict | None) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run_fixture"
        run_dir.mkdir()
        if run_config is not None:
            (run_dir / "run_config.json").write_text(
                json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        materialized = _rewrite_paths(manifest, source=PLACEHOLDER, target=str(run_dir))
        upgraded = _upgrade_manifest_v3(run_dir, materialized)

        case_dir = FIXTURE_ROOT / name
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if run_config is not None:
            (case_dir / "run_config.json").write_text(
                json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        (case_dir / "golden.json").write_text(
            json.dumps(_normalize_output(upgraded, run_dir), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"captured {name}")


def _load_real(run_dir: Path) -> tuple[dict, dict | None]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest = _rewrite_paths(manifest, source=str(run_dir), target=PLACEHOLDER)
    run_config = None
    run_config_path = run_dir / "run_config.json"
    if run_config_path.exists():
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        run_config = _rewrite_paths(run_config, source=str(run_dir), target=PLACEHOLDER)
    return manifest, run_config


def _synthetic_speaker_transcript_case() -> tuple[dict, dict | None]:
    manifest, run_config = _load_real(REAL_RUNS["v2_real_20260910"])
    manifest["speaker_transcript"] = {
        "status": "partial",
        "backend": "segment-clustering",
        "backends": ["segment-clustering"],
        "fallback_used": True,
        "coverage": {"start_ms": 0, "end_ms": 300000, "duration_ms": 300000, "known": True},
        "speakers": [
            {
                "speaker_id": "SPEAKER_00",
                "raw_labels": ["SPEAKER_00"],
                "display_name": "Speaker 1",
                "identity_status": "unknown",
                "confidence": 0.0,
                "identity_evidence": [],
            }
        ],
        "failures": [{"source_label": "001_demo", "error": "demo failure"}],
    }
    manifest["partial_count"] = 1
    return manifest, run_config


def _synthetic_legacy_asr_case() -> tuple[dict, dict | None]:
    manifest = {
        "started_at": "2026-06-01T10:00:00Z",
        "completed_at": "2026-06-01T10:02:30Z",
        "elapsed_seconds": 150,
        "generated_at": "2026-06-01T10:00:00Z",
        "model_name": "ggml-base.bin",
        "language": "zh",
        "results": [
            {
                "source_label": "a_qwen",
                "status": "ready",
                "title": "Item A",
                "asr": {"selected_engine": "qwen3-asr", "requested_model": "Qwen/Qwen3-ASR-0.6B"},
                "output_dir": f"{PLACEHOLDER}/items/a_qwen",
            },
            {
                "source_label": "b_whisper",
                "status": "ready",
                "title": "Item B",
                "asr": {"selected_engine": "whisper-cpp", "fallbacks": [{"engine": "whisper-cpp", "reason": "demo"}]},
                "output_dir": f"{PLACEHOLDER}/items/b_whisper",
            },
        ],
        "error_count": 0,
    }
    run_config = {
        "source_count": 2,
        "started_at": "2026-06-01T10:00:00Z",
        "model_name": "ggml-base.bin",
        "asr": {"requested_engine": "auto"},
        "enhancement": {"postprocess_profile": "balanced"},
    }
    return manifest, run_config


def main() -> None:
    for name, run_dir in REAL_RUNS.items():
        manifest, run_config = _load_real(run_dir)
        _capture_case(name, manifest, run_config)
    _capture_case("v2_speaker_transcript_partial", *_synthetic_speaker_transcript_case())
    _capture_case("v1_legacy_asr_mixed", *_synthetic_legacy_asr_case())


if __name__ == "__main__":
    main()
