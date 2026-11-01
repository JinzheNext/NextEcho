from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import wave

from workbench.diarization_metrics import evaluate_continuous_regions
from workbench.speaker_benchmark import SpeakerRegion as R, evaluate_manifest
from workbench.speaker_scoring_io import dscore_10ms_jer, read_rttm, read_uem


class RecordingScoringTests(unittest.TestCase):
    def fixture(self, root, rid="one", tone=0):
        audio = root / (rid + ".wav")
        with wave.open(str(audio), "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(bytes((tone, 0)) * 16000)
        digest = hashlib.sha256(audio.read_bytes()).hexdigest()
        ref, hyp, uem = [
            root / (rid + suffix) for suffix in (".ref.rttm", ".hyp.rttm", ".uem")
        ]
        ref.write_text(f"SPEAKER {rid} 1 0.003 0.024 <NA> <NA> A <NA> <NA>\n")
        hyp.write_text(f"SPEAKER {rid} 1 0.004 0.024 <NA> <NA> X <NA> <NA>\n")
        uem.write_text(f"{rid} 1 0 0.031\n")

        def annotation(path, kind):
            return {
                "path": path.name,
                "format": "rttm",
                "annotation_kind": kind,
                "annotation_status": "complete",
                "recording_id": rid,
                "source_audio_sha256": digest,
            }

        return {
            "recording_id": rid,
            "source_audio": {"path": audio.name, "sha256": digest},
            "uem": uem.name,
            "reference": annotation(ref, "human-diarization"),
            "hypotheses": {"demo": annotation(hyp, "acoustic-diarization")},
        }

    def score(self, root, items, **options):
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "metric_protocol": "recording-eval-v3",
                    "synthetic_only": True,
                    "items": items,
                    **options,
                }
            )
        )
        return evaluate_manifest(manifest)

    def test_10ms_score_is_separate_from_continuous_and_has_global_mapping(self):
        ref, hyp = [R(3, 27, "A")], [R(4, 28, "X")]
        self.assertEqual(dscore_10ms_jer(ref, hyp, uem_ms=[[0, 31]])["jer"], 0)
        self.assertAlmostEqual(
            evaluate_continuous_regions(ref, hyp, uem_ms=[[0, 31]])["jer"], 0.08
        )
        ref = [R(0, 10, "A"), R(10, 20, "B"), R(30, 40, "A"), R(40, 50, "B")]
        hyp = [R(0, 10, "X"), R(10, 20, "Y"), R(30, 40, "Y"), R(40, 50, "X")]
        result = dscore_10ms_jer(ref, hyp, uem_ms=[[0, 20], [30, 50]])
        self.assertAlmostEqual(result["jer"], 2 / 3)
        self.assertEqual(result["reference_speaker_frames"], {"A": 2, "B": 2})

    def test_zero_sampled_duration_is_not_relabelled_silence_or_fake_zero_error(self):
        zero = dscore_10ms_jer([R(1, 3, "A")], [R(1, 3, "X")], uem_ms=[[0, 10]])
        self.assertIsNone(zero["jer"])
        self.assertFalse(zero["defined"])
        self.assertEqual(zero["reference_zero_frame_labels"], ["A"])
        self.assertEqual(dscore_10ms_jer([], [], uem_ms=[[0, 10]])["jer"], 0)
        self.assertEqual(
            dscore_10ms_jer([R(0, 10, "A")], [], uem_ms=[[0, 10]])["jer"], 1
        )
        self.assertEqual(
            dscore_10ms_jer([], [R(0, 10, "X")], uem_ms=[[0, 10]])["jer"], 1
        )

    def test_labels_wholly_outside_uem_are_trimmed_but_zero_frame_labels_inside_remain(
        self,
    ):
        result = dscore_10ms_jer(
            [R(0, 10, "A"), R(30, 40, "outside")], [R(0, 10, "X")], uem_ms=[[0, 20]]
        )
        self.assertEqual(result["reference_speaker_frames"], {"A": 1})
        self.assertEqual(result["jer"], 0)

    def test_fractional_times_are_explicit_and_old_integer_protocol_is_unchanged(self):
        with self.assertRaises(ValueError):
            evaluate_continuous_regions([R(0.25, 2.75, "A")], [], uem_ms=[[0, 3]])
        result = evaluate_continuous_regions(
            [R(0.25, 2.75, "A")], [], uem_ms=[[0, 3]], allow_fractional_ms=True
        )
        self.assertEqual(result["reference_speaker_ms"], 2.5)
        self.assertEqual(result["metric_protocol"], "continuous-float-ms-v3")

    def test_rttm_uem_preserve_decimal_seconds_exact_ids_and_channels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "a.rttm").write_text(
                "; comment\nSPKR-INFO file.part 2 <NA> <NA> <NA> unknown A <NA>\n"
                "SPEAKER file.part 2 0.00025 0.0025 <NA> <NA> A <NA>\n"
            )
            (root / "a.uem").write_text("# comment\nfile.part 2 0.00025 0.00325\n")
            data, _ = read_rttm(root / "a.rttm")
            row = data["file.part", "2"][0]
            self.assertAlmostEqual(row.start_ms, 0.25)
            self.assertAlmostEqual(row.end_ms, 2.75)
            uem, _ = read_uem(root / "a.uem")
            self.assertEqual(uem["file.part", "2"], [[0.25, 3.25]])

    def test_bad_file_rows_and_nonfinite_boundaries_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad"
            for body in (
                "SPEAKER r 1 NaN 1 <NA> <NA> A <NA>",
                "SPEAKER r 1 0 inf <NA> <NA> A <NA>",
                "SPEAKER r 1 0 0 <NA> <NA> A <NA>",
                "SPEAKER r 1 -1 1 <NA> <NA> A <NA>",
                "NOISE r 1 0 1 <NA> <NA> A <NA>",
                "SPEAKER r 1 0 1",
            ):
                path.write_text(body)
                with self.subTest(body=body), self.assertRaises(ValueError):
                    read_rttm(path)
            for body in ("r 1 0 0", "r 1 NaN 1", "r 1 0 1 extra"):
                path.write_text(body)
                with self.subTest(body=body), self.assertRaises(ValueError):
                    read_uem(path)

    def test_manifest_actual_audio_binding_finite_uem_and_dual_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entry = self.fixture(root)
            result = self.score(root, [entry])
            self.assertAlmostEqual(result["aggregate"]["demo"]["jer_continuous"], 0.08)
            self.assertEqual(result["aggregate"]["demo"]["jer_dscore_10ms"], 0)
            self.assertAlmostEqual(result["aggregate"]["demo"]["der"], 2 / 24)
            entry["source_audio"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "audio hash"):
                self.score(root, [entry])

    def test_collar_and_overlap_apply_to_continuous_but_not_dscore_jer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            entry = self.fixture(root)
            (root / "one.ref.rttm").write_text(
                "SPEAKER one 1 0 .5 <NA> <NA> A <NA>\nSPEAKER one 1 0 .5 <NA> <NA> B <NA>\nSPEAKER one 1 .5 .5 <NA> <NA> C <NA>\n"
            )
            (root / "one.hyp.rttm").write_text("SPEAKER one 1 .5 .5 <NA> <NA> X <NA>\n")
            (root / "one.uem").write_text("one 1 0 1\n")
            excluded = self.score(root, [entry], skip_overlap=True)["aggregate"]["demo"]
            included = self.score(root, [entry])["aggregate"]["demo"]
            self.assertEqual(excluded["jer_continuous"], 0)
            self.assertAlmostEqual(excluded["jer_dscore_10ms"], 2 / 3)
            self.assertEqual(excluded["jer_dscore_10ms"], included["jer_dscore_10ms"])

    def test_pending_empty_wrong_source_channel_and_undeclared_recordings_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = self.fixture(root)
            changes = [
                ("annotation_status", "pending"),
                ("recording_id", "other"),
                ("source_audio_sha256", "0" * 64),
                ("channel", "2"),
            ]
            for key, value in changes:
                entry = json.loads(json.dumps(original))
                entry["reference"][key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    self.score(root, [entry])
            (root / "one.ref.rttm").write_text("")
            with self.assertRaisesRegex(ValueError, "no_speech"):
                self.score(root, [original])
            original["reference"]["content_state"] = "no_speech"
            silent = self.score(root, [original])
            self.assertIsNone(silent["aggregate"]["demo"]["der"])
            (root / "one.ref.rttm").write_text(
                "SPEAKER unknown 1 0 .1 <NA> <NA> A <NA>\n"
            )
            with self.assertRaisesRegex(ValueError, "undeclared"):
                self.score(root, [original])

    def test_one_recording_once_audio_bounds_and_consistent_candidate_set(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            one, two = self.fixture(root), self.fixture(root, "two", 1)
            with self.assertRaisesRegex(ValueError, "occur once"):
                self.score(root, [one, one])
            two["hypotheses"]["other"] = two["hypotheses"].pop("demo")
            with self.assertRaisesRegex(ValueError, "same engine"):
                self.score(root, [one, two])
            (root / "one.uem").write_text("one 1 0 2\n")
            with self.assertRaisesRegex(ValueError, "audio range"):
                self.score(root, [one])
            (root / "one.uem").write_text("one 1 0 1\n")
            (root / "one.hyp.rttm").write_text("SPEAKER one 1 0 2 <NA> <NA> X <NA>\n")
            with self.assertRaisesRegex(ValueError, "actual source"):
                self.score(root, [one])

    def test_existing_cli_consumes_new_manifest_without_models(self):
        from workbench.cli import main

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.score(root, [self.fixture(root)])
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(
                    ["benchmark", "speakers", "--gold-dir", str(root), "--json"]
                )
            self.assertEqual(code, 0)
            self.assertEqual(
                json.loads(out.getvalue())["metric_protocol"], "recording-eval-v3"
            )
            payload = json.loads((root / "manifest.json").read_text())
            payload["items"][0]["reference"]["annotation_status"] = "pending"
            (root / "manifest.json").write_text(json.dumps(payload))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = main(
                    ["benchmark", "speakers", "--gold-dir", str(root), "--json"]
                )
            self.assertEqual(code, 2)
            self.assertEqual(json.loads(out.getvalue())["status"], "unscored")
            self.assertIn("pending is not silence", out.getvalue())


if __name__ == "__main__":
    unittest.main()
