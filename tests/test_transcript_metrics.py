from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path
import random
import tempfile
import unittest
import wave

from workbench.speaker_benchmark import evaluate_manifest
from workbench.transcript_metrics import (
    aggregate_text_scores, edit_distance, evaluate_character_span_attribution, evaluate_text_files,
    normalize_characters, score_characters,
)


def distance_oracle(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def rows(*values):
    return [{"speaker_id": label, "text": text} for label, text in values]


class CharacterMetricTests(unittest.TestCase):
    def test_normalization_preserves_decimal_punctuation_case_and_script(self):
        self.assertEqual(normalize_characters(" Ａ\t３．５\n繁體 "), "A3.5繁體")
        self.assertEqual(score_characters(rows(("a", "3.5")), rows(("x", "35")))["cer"]["errors"], 1)
        self.assertEqual(edit_distance("A體", "a体"), 2)

    def test_bitvector_matches_independent_dynamic_programming(self):
        randomizer = random.Random(73517)
        for _ in range(600):
            a = "".join(randomizer.choices("汉字ab.35", k=randomizer.randrange(45)))
            b = "".join(randomizer.choices("汉字ab.35", k=randomizer.randrange(45)))
            self.assertEqual(edit_distance(a, b), distance_oracle(a, b))
        self.assertEqual(edit_distance("a" * 1000, "b" * 1000), 1000)

    def test_global_cp_assignment_matches_exhaustive_independent_costs(self):
        randomizer = random.Random(719)
        for _ in range(80):
            ref = ["".join(randomizer.choices("甲乙丙丁", k=randomizer.randrange(10))) for _ in range(randomizer.randrange(1, 5))]
            hyp = ["".join(randomizer.choices("甲乙丙丁", k=randomizer.randrange(10))) for _ in range(randomizer.randrange(1, 5))]
            size = max(len(ref), len(hyp))
            padded_r, padded_h = ref + [""] * (size - len(ref)), hyp + [""] * (size - len(hyp))
            expected = min(sum(distance_oracle(padded_r[i], padded_h[j]) for i, j in enumerate(order))
                           for order in itertools.permutations(range(size)))
            result = score_characters(rows(*enumerate_strings(ref, "r")), rows(*enumerate_strings(hyp, "h")))
            self.assertEqual(result["cpcer"]["errors"], expected)

    def test_cp_permutation_is_not_identity_evidence_and_window_swaps_cost(self):
        ref = rows(("A", "甲甲"), ("B", "乙乙"), ("A", "丙丙"), ("B", "丁丁"))
        good = rows(("Y", "甲甲"), ("X", "乙乙"), ("Y", "丙丙"), ("X", "丁丁"))
        swapped = rows(("Y", "甲甲"), ("X", "乙乙"), ("X", "丙丙"), ("Y", "丁丁"))
        self.assertEqual(score_characters(ref, good)["cpcer"]["errors"], 0)
        score = score_characters(ref, swapped)
        self.assertEqual(score["cer"]["errors"], 0)
        self.assertEqual(score["cpcer"]["errors"], 4)
        self.assertFalse(score["cpcer"]["assignment_is_identity_evidence"])

    def test_extra_missing_and_unattributed_streams_are_retained(self):
        result = score_characters(rows(("A", "甲甲")), rows(("X", "甲甲"), (None, "乙乙")))
        self.assertEqual(result["cpcer"]["errors"], 2)
        self.assertEqual(result["unattributed_hypothesis_chars"], 2)
        self.assertTrue(any(x["hypothesis_unattributed"] for x in result["cpcer"]["assignment"]))
        missing = score_characters(rows(("A", "甲"), ("B", "乙")), [])
        self.assertEqual(missing["cpcer"]["errors"], 2)
        empty = score_characters([], rows((None, "乙")))
        self.assertIsNone(empty["cer"]["error_rate"])
        self.assertIsNone(empty["cpcer"]["error_rate"])
        self.assertEqual(empty["cpcer"]["errors"], 1)

    def test_fixed_map_is_distinct_and_requires_explicit_source(self):
        ref, hyp = rows(("A", "甲甲"), ("B", "乙乙")), rows(("X", "乙乙"), ("Y", "甲甲"), (None, "丙"))
        result = score_characters(ref, hyp, fixed_mapping={"X": "A", "Y": "B"}, mapping_source="human-audio-review")
        self.assertEqual(result["cpcer"]["errors"], 1)
        self.assertEqual(result["sa_cer_fixed_map"]["errors"], 5)
        self.assertIsNone(score_characters(ref, hyp)["sa_cer_fixed_map"])
        for mapping, source in [({"X": "A", "Y": "A"}, "human-audio-review"),
                                ({None: "A"}, "human-audio-review"), ({"X": []}, "human-audio-review"),
                                ({"X": "A"}, "cpcer-assignment")]:
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                score_characters(ref, hyp, fixed_mapping=mapping, mapping_source=source)


def enumerate_strings(values, prefix):
    return [(f"{prefix}{i}", text) for i, text in enumerate(values)]


class TextInputTests(unittest.TestCase):
    def fixture(self, root, rid="one", sample=0, text="甲甲"):
        audio = root / f"{rid}.wav"
        with wave.open(str(audio), "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(bytes((sample, 0)) * 16000)
        audio_hash = hashlib.sha256(audio.read_bytes()).hexdigest()
        segment = {"id": "s1", "text": text, "start_ms": 100, "end_ms": 400, "speaker_id": "A"}
        common = {"schema_version": 1, "recording_id": rid, "source_audio_sha256": audio_hash}
        reference = {**common, "annotation_kind": "human-verbatim", "annotation_status": "complete",
                     "text_layer": "verbatim-reference", "segments": [segment]}
        producer = {"timing_precision": "segment", "transcription": [
            {"id": "s1", "text": text, "offsets": {"from": 100, "to": 400}}]}
        self.write(root / f"{rid}.producer.json", producer)
        hypothesis = {**common, "annotation_kind": "hypothesis-transcript", "text_layer": "raw-asr",
                      "segments": [{**segment, "speaker_id": "X"}],
                      "text_source": {"path": f"{rid}.producer.json", "field": "transcription",
                          "sha256": hashlib.sha256((root / f"{rid}.producer.json").read_bytes()).hexdigest(),
                          "timing_policy": "producer-segment-envelopes-v1"}}
        self.write(root / f"{rid}.ref.json", reference)
        self.write(root / f"{rid}.hyp.json", hypothesis)
        for suffix, label in [("ref", "A"), ("hyp", "X")]:
            (root / f"{rid}.{suffix}.rttm").write_text(f"SPEAKER {rid} 1 .1 .3 <NA> <NA> {label} <NA>\n")
        (root / f"{rid}.uem").write_text(f"{rid} 1 0 1\n")
        def descriptor(suffix, kind):
            return {"path": f"{rid}.{suffix}.rttm", "format": "rttm", "annotation_kind": kind,
                    "annotation_status": "complete", "recording_id": rid, "source_audio_sha256": audio_hash}
        return {"recording_id": rid, "source_audio": {"path": audio.name, "sha256": audio_hash},
                "uem": f"{rid}.uem", "reference": descriptor("ref", "human-diarization"),
                "text_reference": {"path": f"{rid}.ref.json", "annotation_kind": "human-verbatim", "annotation_status": "complete"},
                "hypotheses": {"demo": {**descriptor("hyp", "acoustic-diarization"),
                    "text": {"path": f"{rid}.hyp.json", "annotation_kind": "hypothesis-transcript"}}}}

    def write(self, path, payload):
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def score(self, root, entry, **options):
        return evaluate_text_files(entry["text_reference"], entry["hypotheses"]["demo"]["text"],
            root=root, recording_id=entry["recording_id"], source_audio_sha256=entry["source_audio"]["sha256"],
            uem_ms=options.pop("uem_ms", [[0, 1000]]), duration_ms=1000, **options)

    def manifest(self, root, entries):
        path = root / "manifest.json"
        self.write(path, {"schema_version": 3, "metric_protocol": "recording-eval-v3", "synthetic_only": True, "items": entries})
        return evaluate_manifest(path)

    def test_frozen_layer_hash_identity_channel_and_pending_are_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entry = self.fixture(root)
            result = self.score(root, entry)
            self.assertEqual(result["cer"]["errors"], 0)
            self.assertIsNone(result["reference_strata"]["denominators"])
            path = root / "one.hyp.json"
            original = json.loads(path.read_text())
            for key, value in [("source_audio_sha256", "0" * 64), ("recording_id", "other"),
                               ("channel", "2"), ("text_layer", "postprocessed")]:
                bad = copy.deepcopy(original)
                bad[key] = value
                self.write(path, bad)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    self.score(root, entry)
            self.write(path, original)
            entry["text_reference"]["annotation_status"] = "pending"
            with self.assertRaisesRegex(ValueError, "pending"):
                self.score(root, entry)

    def test_uem_never_proportionally_crops_text_or_silently_reorders_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entry = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "crosses UEM"):
                self.score(root, entry, uem_ms=[[0, 250], [300, 1000]])
            result = self.score(root, entry, uem_ms=[[500, 1000]])
            self.assertIsNone(result["cer"]["error_rate"])
            self.assertEqual(result["reference"]["excluded_segment_ids"], ["s1"])
            path = root / "one.hyp.json"
            data = json.loads(path.read_text())
            data["segments"][0]["start_ms"] = 101
            self.write(path, data)
            with self.assertRaisesRegex(ValueError, "timing differs"):
                self.score(root, entry)

    def test_producer_text_and_bytes_cannot_be_replaced_by_enhanced_text(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entry = self.fixture(root)
            path = root / "one.hyp.json"
            data = json.loads(path.read_text())
            data["segments"][0]["text"] = "润色文字"
            self.write(path, data)
            with self.assertRaisesRegex(ValueError, "differs from frozen producer"):
                self.score(root, entry)
            data["segments"][0]["text"] = "甲甲"
            data["text_source"]["sha256"] = "0" * 64
            self.write(path, data)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                self.score(root, entry)

    def test_postprocessed_layer_requires_explicit_task_and_actual_layer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entry = self.fixture(root)
            producer_path, hyp_path = root / "one.producer.json", root / "one.hyp.json"
            producer = json.loads(producer_path.read_text())
            producer["enhanced_transcription"] = producer["transcription"]
            self.write(producer_path, producer)
            hyp = json.loads(hyp_path.read_text())
            hyp["text_layer"] = "postprocessed"
            hyp["text_source"].update(field="enhanced_transcription", sha256=hashlib.sha256(producer_path.read_bytes()).hexdigest())
            self.write(hyp_path, hyp)
            with self.assertRaisesRegex(ValueError, "layer/task"):
                self.score(root, entry)
            self.assertEqual(self.score(root, entry, task="postprocessed")["cer"]["errors"], 0)

    def test_fixed_mapping_file_and_strata_require_complete_frozen_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entry = self.fixture(root)
            path = root / "one.ref.json"
            ref = json.loads(path.read_text())
            ref.update(strata_status="complete", strata_definitions={tag: tag + " definition" for tag in ("short_answer", "overlap", "fragment", "boundary")})
            ref["segments"][0]["tags"] = ["short_answer", "boundary"]
            self.write(path, ref)
            mapping = {"schema_version": 1, "annotation_kind": "fixed-speaker-mapping", "annotation_status": "complete",
                       "recording_id": "one", "source_audio_sha256": entry["source_audio"]["sha256"],
                       "mapping": {"X": "A"}, "mapping_source": "human-audio-review"}
            self.write(root / "map.json", mapping)
            result = self.score(root, entry, fixed_mapping={"path": "map.json"})
            self.assertEqual(result["sa_cer_fixed_map"]["errors"], 0)
            counts = result["reference_strata"]["denominators"]
            self.assertEqual(counts["short_answer"], {"reference_segments": 1, "reference_chars": 2, "reference_speaker_ms": 300})
            self.assertEqual(counts["overlap"]["reference_chars"], 0)
            ref["segments"][0]["tags"] = [{}]
            self.write(path, ref)
            with self.assertRaisesRegex(ValueError, "tags list"):
                self.score(root, entry)

    def test_empty_pending_and_duplicate_json_keys_are_not_zero_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entry = self.fixture(root)
            path = root / "one.ref.json"
            original = path.read_text()
            ref = json.loads(original)
            ref["segments"] = []
            self.write(path, ref)
            with self.assertRaisesRegex(ValueError, "no_speech"):
                self.score(root, entry)
            path.write_text(original[:-1] + ',"annotation_status":"complete"}')
            with self.assertRaisesRegex(ValueError, "duplicate JSON"):
                self.score(root, entry)

    def test_real_manifest_integrates_all_engines_and_micro_averages(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            one, two = self.fixture(root), self.fixture(root, "two", 1, "乙乙乙乙")
            refpath = root / "two.ref.json"
            ref = json.loads(refpath.read_text())
            ref["segments"][0]["text"] = "乙甲乙乙"
            self.write(refpath, ref)
            result = self.manifest(root, [one, two])
            self.assertEqual(result["text_status"], "scored")
            score = result["aggregate"]["demo"]["text_metrics"]
            self.assertEqual(score["cer"]["reference_chars"], 6)
            self.assertEqual(score["cer"]["errors"], 1)
            self.assertAlmostEqual(score["cer"]["error_rate"], 1 / 6)
            del two["text_reference"]
            with self.assertRaisesRegex(ValueError, "every recording"):
                self.manifest(root, [one, two])
            two["text_reference"] = {**one["text_reference"], "path": "two.ref.json"}
            del two["hypotheses"]["demo"]["text"]
            with self.assertRaisesRegex(ValueError, "annotation declaration"):
                self.manifest(root, [one, two])

    def test_text_labels_must_exist_in_frozen_acoustic_prediction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entry = self.fixture(root)
            path = root / "one.hyp.json"
            hyp = json.loads(path.read_text())
            hyp["segments"][0]["speaker_id"] = "invented"
            self.write(path, hyp)
            with self.assertRaisesRegex(ValueError, "acoustic labels"):
                self.manifest(root, [entry])

    def test_sparse_uem_uses_one_character_assignment_for_entire_recording(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entry = self.fixture(root)
            ref_path, hyp_path, source_path = (root / f"one.{s}.json" for s in ("ref", "hyp", "producer"))
            ref, hyp, source = (json.loads(p.read_text()) for p in (ref_path, hyp_path, source_path))
            ref["segments"], hyp["segments"], source["transcription"] = [], [], []
            for i, (start, end, label, guessed, text) in enumerate([
                (0, 100, "A", "X", "甲甲"), (100, 200, "B", "Y", "乙乙"),
                (600, 700, "A", "Y", "丙丙"), (700, 800, "B", "X", "丁丁")]):
                segment = {"id": f"s{i}", "start_ms": start, "end_ms": end, "text": text}
                ref["segments"].append({**segment, "speaker_id": label})
                hyp["segments"].append({**segment, "speaker_id": guessed})
                source["transcription"].append({"id": f"s{i}", "text": text, "offsets": {"from": start, "to": end}})
            self.write(source_path, source)
            hyp["text_source"]["sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
            self.write(ref_path, ref)
            self.write(hyp_path, hyp)
            score = self.score(root, entry, uem_ms=[[0, 200], [600, 800]])
            self.assertEqual(score["cer"]["errors"], 0)
            self.assertEqual(score["cpcer"]["errors"], 4)
            self.assertEqual(score["cpcer"]["mapping_scope"], "recording")

    def test_invalid_labels_rows_or_fixed_mapping_are_value_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entry = self.fixture(root)
            path = root / "one.ref.json"
            original = json.loads(path.read_text())
            for value in (None, [], ""):
                data = copy.deepcopy(original)
                data["segments"][0]["speaker_id"] = value
                self.write(path, data)
                with self.subTest(value=value), self.assertRaises(ValueError):
                    self.manifest(root, [entry])
            self.write(path, original)
            with self.assertRaises(ValueError):
                self.score(root, entry, task=[])

    def test_aggregate_never_silently_mixes_fixed_mapping_conditions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entry = self.fixture(root)
            one = self.score(root, entry)
            two = copy.deepcopy(one)
            two["sa_cer_fixed_map"] = {**two["cer"], "mapping_source": "human-audio-review"}
            self.assertIsNone(aggregate_text_scores([one, two])["sa_cer_fixed_map"])
            one["sa_cer_fixed_map"] = {**one["cer"], "mapping_source": "time-overlap-oracle"}
            self.assertIsNone(aggregate_text_scores([one, two])["sa_cer_fixed_map"])
            two["sa_cer_fixed_map"]["mapping_source"] = "time-overlap-oracle"
            self.assertEqual(aggregate_text_scores([one, two])["sa_cer_fixed_map"]["mapping_source"], "time-overlap-oracle")

    def test_native_character_spans_score_speaker_assignment_and_report_zero_width(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            text = "甲乙"
            source = root / "transcript.enhanced.json"
            self.write(source, {"enhanced_transcription": [{"id": "s", "text": text,
                "offsets": {"from": 0, "to": 400}}]})
            audio = root / "audio.wav"
            audio.write_bytes(b"synthetic audio source")
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            audio_hash = hashlib.sha256(audio.read_bytes()).hexdigest()
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            hypothesis = root / "transcript.identity.json"
            self.write(hypothesis, {"schema_version": 2, "annotation_kind": "speaker-attributed-character-spans",
                "recording_id": "one", "source_audio": str(audio), "source_audio_sha256": audio_hash,
                "source_path": str(source), "source_sha256": source_hash, "source_field": "enhanced_transcription",
                "text_reconstruction_exact": True, "status": "partial", "transcription": [
                    {"id": "s-voice-0", "parent_segment_id": "s", "text": "甲", "text_start": 0, "text_end": 1,
                     "source_text_sha256": text_hash, "speaker_id": "spk_00", "voice_evidence": [
                         {"text_start": 0, "text_end": 1, "source_text_sha256": text_hash,
                          "timing_kind": "native_interval", "native_start_ms": 100, "native_end_ms": 200}]},
                    {"id": "s-voice-1", "parent_segment_id": "s", "text": "乙", "text_start": 1, "text_end": 2,
                     "source_text_sha256": text_hash, "speaker_id": "", "voice_evidence": [
                         {"text_start": 1, "text_end": 2, "source_text_sha256": text_hash,
                          "timing_kind": "invalid_zero_width", "native_start_ms": 200, "native_end_ms": 200}]},
                ]})
            result = evaluate_character_span_attribution(hypothesis, [
                {"speaker_id": "A", "start_ms": 100, "end_ms": 200},
                {"speaker_id": "B", "start_ms": 200, "end_ms": 300},
            ], root=root, recording_id="one", source_audio_sha256=audio_hash,
                uem_ms=[[0, 400]], duration_ms=1000)
            self.assertEqual(result["metric_protocol"], "speaker-character-agreement-v1")
            self.assertEqual(result["counts"]["speaker_accuracy"], 1.0)
            self.assertEqual(result["counts"]["scored_chars"], 1)
            self.assertEqual(result["counts"]["invalid_zero_width_chars"], 1)
            self.assertFalse(result["model_release_approved"])


if __name__ == "__main__":
    unittest.main()
