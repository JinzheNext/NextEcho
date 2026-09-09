from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from workbench.speaker_benchmark import SpeakerRegion, evaluate_manifest, evaluate_regions


class SpeakerBenchmarkTests(unittest.TestCase):
    def test_permutation_invariant_perfect_match(self) -> None:
        reference = [
            SpeakerRegion(0, 1000, "孟岩"),
            SpeakerRegion(1000, 2000, "李继刚"),
            SpeakerRegion(2000, 3000, "孟岩"),
        ]
        hypothesis = [
            SpeakerRegion(0, 1000, "SPEAKER_01"),
            SpeakerRegion(1000, 2000, "SPEAKER_00"),
            SpeakerRegion(2000, 3000, "SPEAKER_01"),
        ]
        result = evaluate_regions(reference, hypothesis, frame_ms=20)
        self.assertEqual(result["der"], 0)
        self.assertEqual(result["jer"], 0)
        self.assertEqual(result["false_switches"], 0)
        self.assertEqual(result["speaker_mapping"], {"SPEAKER_00": "李继刚", "SPEAKER_01": "孟岩"})

    def test_reports_confusion_and_false_switch(self) -> None:
        reference = [SpeakerRegion(0, 2000, "孟岩")]
        hypothesis = [SpeakerRegion(0, 1000, "S0"), SpeakerRegion(1000, 2000, "S1")]
        result = evaluate_regions(reference, hypothesis, frame_ms=20)
        self.assertEqual(result["confusion_ms"], 1000)
        self.assertEqual(result["der"], 0.5)
        self.assertEqual(result["false_switches"], 1)

    def test_manifest_paths_are_relative_to_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "gold.json").write_text(json.dumps({"segments": [{"start_ms": 0, "end_ms": 1000, "speaker_id": "A"}]}), encoding="utf-8")
            (root / "hyp.json").write_text(json.dumps({"speaker_turns": [{"start": 0, "end": 1, "speaker": "Speaker 1"}]}), encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"items": [{"id": "sample", "reference": "gold.json", "hypotheses": {"demo": "hyp.json"}}]}), encoding="utf-8")
            result = evaluate_manifest(manifest)
        self.assertEqual(result["aggregate"]["demo"]["der"], 0)
        self.assertEqual(result["aggregate"]["demo"]["items"], 1)


if __name__ == "__main__":
    unittest.main()
