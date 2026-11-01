"""Golden contract tests for the manifest v3 upgrade.

These tests lock the exact behavior of the manifest upgrade pipeline against
fixtures captured from the baseline implementation (see
``tests/fixtures/manifest_upgrade/capture.py``).  The fixtures include real
historical v1/v2 manifests from ``outputs/transcriptions`` plus synthetic
branch cases, so any refactor of the upgrade logic must reproduce them field
for field.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app import _upgrade_manifest_v3

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "manifest_upgrade"
PLACEHOLDER = "<RUN_DIR>"
VOLATILE_KEYS = {"captured_at", "peak_rss_bytes", "peak_rss_delta_bytes"}


def _rewrite_paths(value: object, *, source: str, target: str) -> object:
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


def _normalize(manifest: dict, run_dir: Path) -> dict:
    for prefix in {str(run_dir), str(run_dir.resolve())}:
        manifest = _rewrite_paths(manifest, source=prefix, target=PLACEHOLDER)
    return _strip_volatile(manifest)


class ManifestUpgradeGoldenTests(unittest.TestCase):
    def test_upgrade_matches_golden_fixtures(self) -> None:
        cases = sorted(path.name for path in FIXTURE_ROOT.iterdir() if path.is_dir())
        self.assertGreaterEqual(len(cases), 4, "golden fixture cases went missing")
        for case in cases:
            with self.subTest(case=case):
                manifest = json.loads((FIXTURE_ROOT / case / "manifest.json").read_text(encoding="utf-8"))
                run_config_path = FIXTURE_ROOT / case / "run_config.json"
                golden = json.loads((FIXTURE_ROOT / case / "golden.json").read_text(encoding="utf-8"))
                with tempfile.TemporaryDirectory() as tmp:
                    run_dir = Path(tmp) / "run_fixture"
                    run_dir.mkdir()
                    if run_config_path.exists():
                        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
                        run_config = _rewrite_paths(run_config, source=PLACEHOLDER, target=str(run_dir))
                        (run_dir / "run_config.json").write_text(
                            json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
                        )
                    materialized = _rewrite_paths(manifest, source=PLACEHOLDER, target=str(run_dir))
                    upgraded = _upgrade_manifest_v3(run_dir, materialized)
                    self.assertEqual(_normalize(upgraded, run_dir), golden)

    def test_upgrade_is_idempotent(self) -> None:
        # Running the upgrade twice must not drift: serialize_run applies it on
        # every read, so v3 payloads pass through the same pipeline.
        for case in ("v1_real_20260613", "v2_real_20260910"):
            with self.subTest(case=case):
                manifest = json.loads((FIXTURE_ROOT / case / "manifest.json").read_text(encoding="utf-8"))
                run_config_path = FIXTURE_ROOT / case / "run_config.json"
                with tempfile.TemporaryDirectory() as tmp:
                    run_dir = Path(tmp) / "run_fixture"
                    run_dir.mkdir()
                    if run_config_path.exists():
                        (run_dir / "run_config.json").write_text(run_config_path.read_text(encoding="utf-8"))
                    materialized = _rewrite_paths(manifest, source=PLACEHOLDER, target=str(run_dir))
                    once = _upgrade_manifest_v3(run_dir, materialized)
                    twice = _upgrade_manifest_v3(run_dir, once)
                    self.assertEqual(_normalize(once, run_dir), _normalize(twice, run_dir))


if __name__ == "__main__":
    unittest.main()
