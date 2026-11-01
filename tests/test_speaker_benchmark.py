from __future__ import annotations

import json
import itertools
import random
import tempfile
import unittest
from pathlib import Path

from workbench.speaker_benchmark import SpeakerRegion, evaluate_manifest, evaluate_regions, optimal_speaker_mapping


class SpeakerBenchmarkTests(unittest.TestCase):
    @staticmethod
    def mapping_score(weights: list[list[int]]) -> tuple[int, dict[str, str]]:
        reference: list[set[str]] = []
        hypothesis: list[set[str]] = []
        for hyp, row in enumerate(weights):
            for ref, count in enumerate(row):
                reference.extend([{f"R{ref}"}] * count)
                hypothesis.extend([{f"H{hyp}"}] * count)
        # Include silent/unmatched labels so zero rows and columns are tested.
        for ref in range(len(weights[0])):
            reference.append({f"R{ref}"})
            hypothesis.append(set())
        for hyp in range(len(weights)):
            reference.append(set())
            hypothesis.append({f"H{hyp}"})
        mapping = optimal_speaker_mapping(reference, hypothesis)
        score = sum(weights[int(hyp[1:])][int(ref[1:])] for hyp, ref in mapping.items())
        return score, mapping

    def test_more_than_eight_clusters_uses_global_optimum(self) -> None:
        weights = [[0] * 9 for _ in range(9)]
        weights[0][:2] = [6, 5]
        weights[1][0] = 4
        score, mapping = self.mapping_score(weights)
        self.assertEqual(score, 9)  # Sequential greedy took H0->R0: only 6.
        self.assertEqual(mapping, {"H0": "R1", "H1": "R0"})

    def test_rectangular_assignment_matches_brute_force_oracle(self) -> None:
        rng = random.Random(20261019)
        for rows, columns in [(1, 1), (2, 5), (5, 2), (3, 4), (4, 3), (6, 6)]:
            for _ in range(12):
                weights = [[rng.randrange(8) for _ in range(columns)] for _ in range(rows)]
                if rows <= columns:
                    expected = max(sum(weights[r][c] for r, c in enumerate(p)) for p in itertools.permutations(range(columns), rows))
                else:
                    expected = max(sum(weights[r][c] for c, r in enumerate(p)) for p in itertools.permutations(range(rows), columns))
                score, mapping = self.mapping_score(weights)
                with self.subTest(shape=(rows, columns), weights=weights):
                    self.assertEqual(score, expected)
                    self.assertEqual(len(set(mapping.values())), len(mapping))

    def test_many_unmatched_clusters_and_ties_are_deterministic(self) -> None:
        weights = [[0, 0] for _ in range(154)]
        weights[0] = [6, 5]
        weights[153] = [4, 0]
        self.assertEqual(self.mapping_score(weights), (9, {"H0": "R1", "H153": "R0"}))
        tied = [[1, 1], [1, 1], [0, 0]]
        first = self.mapping_score(tied)
        self.assertEqual(first[0], 2)
        for _ in range(5):
            self.assertEqual(self.mapping_score(tied), first)
        self.assertEqual(optimal_speaker_mapping([], []), {})

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
