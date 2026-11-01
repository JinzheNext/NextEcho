from __future__ import annotations

import itertools
import json
import random
import tempfile
import unittest
from pathlib import Path

from workbench.diarization_metrics import evaluate_continuous_regions
from workbench.speaker_benchmark import SpeakerRegion as R, evaluate_manifest, load_regions


class ContinuousSpeakerMetricTests(unittest.TestCase):
    def test_explicit_uem_counts_false_alarm_outside_reference_extent(self) -> None:
        result = evaluate_continuous_regions([R(1000, 2000, "A")], [R(0, 3000, "X")], uem_ms=[[0, 3000]])
        self.assertEqual(result["false_alarm_ms"], 2000)
        self.assertEqual(result["reference_speaker_ms"], 1000)
        self.assertEqual(result["der"], 2)
        self.assertAlmostEqual(result["jer"], 2 / 3)
        cropped = evaluate_continuous_regions([R(1000, 2000, "A")], [R(0, 3000, "X")], uem_ms=[[1000, 2000]])
        self.assertEqual(cropped["der"], 0)

    def test_non_grid_boundaries_and_tail_are_exact(self) -> None:
        result = evaluate_continuous_regions([R(3, 27, "A")], [R(4, 28, "X")], uem_ms=[[0, 31]])
        self.assertEqual(result["correct_speaker_ms"], 23)
        self.assertEqual(result["miss_ms"], 1)
        self.assertEqual(result["false_alarm_ms"], 1)
        self.assertEqual(result["reference_speaker_ms"], 24)
        self.assertAlmostEqual(result["der"], 2 / 24)
        self.assertAlmostEqual(result["jer"], 2 / 25)

    def test_overlap_duplicates_uem_gaps_and_collar(self) -> None:
        reference = [R(0, 1000, "A"), R(500, 1500, "B"), R(600, 700, "B")]
        hypothesis = [R(0, 1500, "X")]
        overlap = evaluate_continuous_regions(reference, hypothesis, uem_ms=[[0, 1500]])
        self.assertEqual(overlap["reference_speaker_ms"], 2000)
        self.assertEqual(overlap["miss_ms"], 500)
        self.assertEqual(overlap["confusion_ms"], 500)
        exclusive = evaluate_continuous_regions(reference, hypothesis, uem_ms=[[0, 1500]], skip_overlap=True)
        self.assertEqual(exclusive["reference_speaker_ms"], 1000)
        self.assertEqual(exclusive["miss_ms"], 0)
        self.assertEqual(exclusive["confusion_ms"], 500)
        gaps = evaluate_continuous_regions([R(0, 1000, "A")], [R(0, 1000, "X")], uem_ms=[[0, 100], [50, 200], [800, 1000]])
        self.assertEqual(gaps["reference_speaker_ms"], 400)
        self.assertEqual(gaps["uem_ms"], [[0, 200], [800, 1000]])
        collar = evaluate_continuous_regions([R(100, 900, "A")], [R(120, 880, "X")], uem_ms=[[0, 1000]], collar_ms=100)
        self.assertEqual(collar["scored_uem_ms"], [[0, 50], [150, 850], [950, 1000]])
        self.assertEqual(collar["reference_speaker_ms"], 700)
        self.assertEqual(collar["der"], 0)
        half_ms = evaluate_continuous_regions([R(1, 3, "A")], [], uem_ms=[[0, 4]], collar_ms=1)
        self.assertEqual(half_ms["reference_speaker_ms"], 1)

    def test_one_recording_mapping_exposes_cross_window_id_swap(self) -> None:
        result = evaluate_continuous_regions(
            [R(0, 10, "A"), R(10, 20, "B"), R(30, 40, "A"), R(40, 50, "B")],
            [R(0, 10, "X"), R(10, 20, "Y"), R(30, 40, "Y"), R(40, 50, "X")],
            uem_ms=[[0, 20], [30, 50]],
        )
        self.assertEqual(result["confusion_ms"], 20)
        self.assertEqual(result["der"], 0.5)

    def test_der_mapping_uses_full_uem_before_exclusions_like_md_eval(self) -> None:
        result = evaluate_continuous_regions(
            [R(0, 1000, "A"), R(0, 1000, "C"), R(1000, 1500, "B")],
            [R(0, 1500, "X")], uem_ms=[[0, 1500]], skip_overlap=True,
        )
        self.assertEqual(result["confusion_ms"], 500)
        self.assertEqual(result["der"], 1)
        self.assertEqual(result["jer"], 0)
        self.assertEqual(result["jer_speaker_mapping"], {"X": "B"})

    def test_jer_has_independent_iou_assignment(self) -> None:
        # Co-occurrence ties; deterministic DER chooses A, IoU chooses B.
        result = evaluate_continuous_regions(
            [R(0, 1000, "A"), R(0, 100, "B")], [R(0, 100, "X")], uem_ms=[[0, 1000]],
        )
        # Equal co-occurrence: deterministic DER picks A, independent IoU picks B.
        self.assertEqual(result["speaker_mapping"], {"X": "A"})
        self.assertEqual(result["jer_speaker_mapping"], {"X": "B"})
        self.assertEqual(result["jer"], 0.5)

    def test_silence_miss_and_invalid_protocol_inputs(self) -> None:
        silent = evaluate_continuous_regions([], [], uem_ms=[[0, 1000]])
        self.assertEqual((silent["der"], silent["jer"], silent["der_defined"]), (None, 0, False))
        false_alarm = evaluate_continuous_regions([], [R(0, 1000, "X")], uem_ms=[[0, 1000]])
        self.assertEqual((false_alarm["false_alarm_ms"], false_alarm["der"], false_alarm["jer"]), (1000, None, 1))
        missed = evaluate_continuous_regions([R(0, 1000, "A")], [], uem_ms=[[0, 1000]])
        self.assertEqual((missed["miss_ms"], missed["der"], missed["jer"]), (1000, 1, 1))
        for uem in (None, [], [[1, 1]], [[-1, 3]], [[0, 1.5]], [[False, 3]]):
            with self.subTest(uem=uem), self.assertRaises(ValueError):
                evaluate_continuous_regions([], [], uem_ms=uem)
        for kwargs in ({"collar_ms": -1}, {"collar_ms": True}, {"skip_overlap": "false"}):
            with self.assertRaises(ValueError):
                evaluate_continuous_regions([], [], uem_ms=[[0, 10]], **kwargs)

    def test_random_integer_intervals_match_millisecond_brute_force(self) -> None:
        rng = random.Random(493)
        for case in range(50):
            reference = [R(start, rng.randrange(start + 1, 42), label) for start, label in ((rng.randrange(30), "A"), (rng.randrange(30), "B"))]
            hypothesis = [R(start, rng.randrange(start + 1, 42), label) for start, label in ((rng.randrange(30), "X"), (rng.randrange(30), "Y"), (rng.randrange(30), "Z"))]
            rs = [{r.speaker for r in reference if r.start_ms <= t < r.end_ms} for t in range(42)]
            hs = [{r.speaker for r in hypothesis if r.start_ms <= t < r.end_ms} for t in range(42)]
            refs = sorted(set.union(*rs))
            hyps = sorted(set.union(*hs))
            weights = {(h, r): sum(h in hset and r in rset for rset, hset in zip(rs, hs)) for h in hyps for r in refs}
            assignments = [dict(zip(perm, refs)) for perm in itertools.permutations(hyps, len(refs))]
            correct = max(sum(weights[h, r] for h, r in mapping.items()) for mapping in assignments)
            ref_total = sum(map(len, rs))
            miss = sum(max(len(r) - len(h), 0) for r, h in zip(rs, hs))
            fa = sum(max(len(h) - len(r), 0) for r, h in zip(rs, hs))
            confusion = sum(min(len(r), len(h)) for r, h in zip(rs, hs)) - correct
            best_iou = max(sum(weights[h, r] / sum(r in rset or h in hset for rset, hset in zip(rs, hs)) for h, r in mapping.items()) for mapping in assignments)
            actual = evaluate_continuous_regions(reference, hypothesis, uem_ms=[[0, 42]])
            with self.subTest(case=case):
                self.assertEqual(actual["reference_speaker_ms"], ref_total)
                self.assertEqual((actual["miss_ms"], actual["false_alarm_ms"], actual["confusion_ms"]), (miss, fa, confusion))
                self.assertAlmostEqual(actual["jer"], 1 - best_iou / len(refs))

    def test_versioned_manifest_freezes_inputs_and_aggregates_speaker_jer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def save(name, kind, rows, recording="one", audio_hash="a" * 64):
                (root / name).write_text(json.dumps({"annotation_kind": kind, "recording_id": recording, "source_audio_sha256": audio_hash, "speaker_turns": rows}), encoding="utf-8")
                return {"path": name, "annotation_kind": kind}
            ref = save("ref.json", "human-diarization", [{"start_ms": 0, "end_ms": 25, "speaker_id": "A"}])
            hyp = save("hyp.json", "acoustic-diarization", [{"start_ms": 0, "end_ms": 30, "speaker_id": "X"}])
            payload = {"schema_version": 2, "metric_protocol": "continuous-ms-v2", "items": [{"id": "one", "recording_id": "one", "uem_ms": [[0, 30]], "reference": ref, "hypotheses": {"demo": hyp}}]}
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            result = evaluate_manifest(manifest)
            self.assertEqual(result["aggregate"]["demo"]["false_alarm_ms"], 5)
            self.assertAlmostEqual(result["aggregate"]["demo"]["der"], 0.2)
            self.assertEqual(len(result["items"][0]["hypothesis_sha256"]), 64)
            # A second recording has two missed speakers. Aggregate JER weights
            # all three reference speakers, not the mean of two file scores.
            second_ref = save("ref2.json", "human-diarization", [
                {"start_ms": 0, "end_ms": 25, "speaker_id": "A"},
                {"start_ms": 0, "end_ms": 25, "speaker_id": "B"},
            ], recording="two", audio_hash="b" * 64)
            empty_hyp = save("hyp2.json", "acoustic-diarization", [], recording="two", audio_hash="b" * 64)
            payload["items"].append({"id": "two", "recording_id": "two", "uem_ms": [[0, 30]], "reference": second_ref, "hypotheses": {"demo": empty_hyp}})
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            aggregate = evaluate_manifest(manifest)["aggregate"]["demo"]
            self.assertAlmostEqual(aggregate["jer"], (1 / 6 + 2) / 3)
            self.assertAlmostEqual(aggregate["der"], 55 / 75)
            original = json.loads(json.dumps(payload))
            payload["items"].append({**payload["items"][0], "id": "one-again"})
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "recording_id"):
                evaluate_manifest(manifest)
            payload = original
            corrupted = json.loads((root / "hyp.json").read_text(encoding="utf-8"))
            corrupted["source_audio_sha256"] = "c" * 64
            (root / "hyp.json").write_text(json.dumps(corrupted), encoding="utf-8")
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "same source audio"):
                evaluate_manifest(manifest)
            corrupted["source_audio_sha256"] = "a" * 64
            (root / "hyp.json").write_text(json.dumps(corrupted), encoding="utf-8")
            # Never accept a text-derived speaker track as acoustic gold/prediction.
            payload["items"][0]["hypotheses"]["demo"]["annotation_kind"] = "text-derived"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluate_manifest(manifest)

    def test_strict_annotation_rejects_malformed_rows_instead_of_silence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotation.json"
            invalid = [
                {}, {"speaker_turns": "bad"}, {"speaker_turns": [None]},
                {"speaker_turns": [{"start_ms": None, "end_ms": 10, "speaker_id": "A"}]},
                {"speaker_turns": [{"start_ms": True, "end_ms": 10, "speaker_id": "A"}]},
                {"speaker_turns": [{"start_ms": 0.5, "end_ms": 10, "speaker_id": "A"}]},
                {"speaker_turns": [{"start_ms": 0, "end_ms": 0, "speaker_id": "A"}]},
                {"speaker_turns": [{"start_ms": 0, "end_ms": 10, "speaker_id": ["A"]}]},
            ]
            for payload in invalid:
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    load_regions(path, strict=True)
            path.write_text('{"speaker_turns": []}', encoding="utf-8")
            self.assertEqual(load_regions(path, strict=True), [])


if __name__ == "__main__":
    unittest.main()
