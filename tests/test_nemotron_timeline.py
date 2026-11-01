from __future__ import annotations

import array
import copy
import ctypes
import json
from pathlib import Path
import tempfile
import unittest

from workbench.contracts import PipelineError
from workbench.io_utils import file_hash, write_json
from workbench.nemotron_candidate import validate_candidate
from workbench.nemotron_timeline import segment_retained_probabilities, verify_long_timeline


class NemotronTimelineTests(unittest.TestCase):
    def fixture(self, folder):
        # Deliberately tiny synthetic compaction clock, not a model/gold run.
        values = array.array("f", [0.] * 101 * 8)
        for frame in range(20, 80):
            values[frame * 8 + 1] = .9
        path = folder / "tail.f32"
        path.write_bytes(values.tobytes())
        prefix = {"start": .1, "end": .7, "speaker": 1}
        tail = segment_retained_probabilities(values, first_frame=100)
        segments = [prefix, *tail]
        checkpoints = []
        for i, (count, first, samples, rows) in enumerate([
            (120, 0, 24000, [prefix]), (150, 100, 24000, [prefix]), (201, 100, 32000, segments)]):
            p = folder / f"{i}.json"
            write_json(p, {"frame_count": count, "retained_frame_start": first,
                           "input_samples_consumed": samples, "segments": rows})
            checkpoints.append({"path": str(p), "sha256": file_hash(p)})
        return {"schema_version": 2, "artifact_kind": "nemotron-native-candidate", "frame_count": 201,
                "speaker_slot_capacity": 8, "seconds_per_frame": .01, "retained_frame_start": 100,
                "origin_ms": 10000, "duration_seconds": 2, "backend_requested": "metal", "preset": "v3-offline",
                "full_attention": False, "probability_path": str(path), "probability_sha256": file_hash(path),
                "segments": segments, "timeline_protocol": "native-retained-tail-v2", "completed": True,
                "input_samples_consumed": 32000, "probability_scope": "retained-tail",
                "segments_scope": "native-whole-timeline", "resume_granularity": "whole-stage", "checkpoints": checkpoints}, values

    def test_tail_bytes_global_clock_and_final_frozen_prefix_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp).resolve()
            data, values = self.fixture(folder)
            self.assertIs(validate_candidate(data, folder, duration=2, origin_ms=10000, long_context=True), data)
            result = verify_long_timeline(data, folder, values)
            self.assertTrue(result["compaction_observed"])
            self.assertEqual(result["retained_probability_frames"], 101)
            self.assertEqual(result["probability_tail_segments_compared"], 1)
            with self.assertRaises(PipelineError):
                validate_candidate(data, folder, duration=2, origin_ms=10000)

    def test_tail_local_times_or_lost_frozen_prefix_cannot_look_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp).resolve()
            data, values = self.fixture(folder)
            local = copy.deepcopy(data)
            local["segments"] = [{**r, "start": r["start"] - 1, "end": r["end"] - 1} for r in data["segments"][1:]]
            with self.assertRaisesRegex(ValueError, "binarization/global clock"):
                verify_long_timeline(local, folder, values)
            lost = {**data, "segments": data["segments"][1:]}
            with self.assertRaisesRegex(ValueError, "frozen prefix"):
                verify_long_timeline(lost, folder, values)

    def test_incomplete_frame_or_sample_coverage_and_missing_final_checkpoint_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp).resolve()
            data, values = self.fixture(folder)
            for change in ({"completed": False}, {"input_samples_consumed": 31999}, {"frame_count": 50},
                           {"probability_scope": "full-audio"}, {"checkpoints": data["checkpoints"][:-1]}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    verify_long_timeline({**data, **change}, folder, values)

    def test_checkpoint_tampering_or_clock_reversal_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp).resolve()
            data, values = self.fixture(folder)
            p = Path(data["checkpoints"][1]["path"])
            checkpoint = json.loads(p.read_text())
            checkpoint["frame_count"] = 1
            write_json(p, checkpoint)
            with self.assertRaisesRegex(ValueError, "changed"):
                verify_long_timeline(data, folder, values)
            data["checkpoints"][1]["sha256"] = file_hash(p)
            with self.assertRaisesRegex(ValueError, "clock"):
                verify_long_timeline(data, folder, values)

    def test_native_hysteresis_thresholds_padding_merge_and_short_drop_are_distinct(self):
        values = array.array("f", [0.] * 200 * 8)
        # Equality to native float onset does not open speech.
        for frame in range(60):
            values[frame * 8] = ctypes.c_float(.641).value
        self.assertEqual(segment_retained_probabilities(values, first_frame=0), [])
        for a, b in [(0, 60), (90, 150)]:
            for frame in range(a, b):
                values[frame * 8] = .9
        result = segment_retained_probabilities(values, first_frame=100)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["start"], 1.)
        self.assertAlmostEqual(result[0]["end"], 2.579)
        one = array.array("f", [0.] * 100 * 8)
        one[40 * 8] = .9
        self.assertEqual(segment_retained_probabilities(one, first_frame=0), [])

    def test_extra_full_timeline_probability_bytes_are_rejected_after_compaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp).resolve()
            data, _ = self.fixture(folder)
            path = Path(data["probability_path"])
            path.write_bytes(array.array("f", [0.] * 201 * 8).tobytes())
            data["probability_sha256"] = file_hash(path)
            with self.assertRaises(PipelineError):
                validate_candidate(data, folder, duration=2, origin_ms=10000, long_context=True)

    def test_native_float_cadence_is_preserved_across_long_global_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp).resolve()
            data, values = self.fixture(folder)
            cadence = ctypes.c_float(.01).value
            data.update(frame_count=50101, retained_frame_start=50000, duration_seconds=501,
                        input_samples_consumed=501 * 16000, seconds_per_frame=cadence)
            data["segments"] = [{"start": .1, "end": .7, "speaker": 1},
                *segment_retained_probabilities(values, first_frame=50000, cadence=cadence)]
            checkpoint = folder / "native-clock-final.json"
            write_json(checkpoint, {"frame_count": data["frame_count"], "retained_frame_start": 50000,
                "input_samples_consumed": data["input_samples_consumed"], "segments": data["segments"]})
            data["checkpoints"] = [{"path": str(checkpoint), "sha256": file_hash(checkpoint)}]
            result = verify_long_timeline(data, folder, values)
            self.assertEqual(result["probability_tail_segments_compared"], 1)
            wrong_clock = {**data, "seconds_per_frame": .01}
            with self.assertRaisesRegex(ValueError, "binarization/global clock|compaction cut"):
                verify_long_timeline(wrong_clock, folder, values)


if __name__ == "__main__":
    unittest.main()
