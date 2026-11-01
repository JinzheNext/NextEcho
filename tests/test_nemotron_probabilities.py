from __future__ import annotations

import array
import copy
from pathlib import Path
import tempfile
import unittest

from workbench.io_utils import file_hash
from workbench.nemotron_probabilities import ProbabilityArchive, verify_full_probability_archive
from workbench.nemotron_timeline import segment_retained_probabilities


class FullProbabilityTests(unittest.TestCase):
    def fixture(self, root):
        values = array.array("f", [0.] * 201 * 8)
        for f in range(70):
            values[f * 8] = .9
        for f in range(140, 201):
            values[f * 8 + 1] = .9
        raw = values.tobytes()
        archive = ProbabilityArchive(root / "whole.f32")
        for first, count in [(0, 100), (50, 150), (100, 201)]:
            archive.update(first_frame=first, frame_count=count, retained_bytes=raw[first*32:count*32])
        metadata = archive.finish(frame_count=201)
        tail = root / "tail.f32"
        tail.write_bytes(raw[100*32:])
        return {"frame_count": 201, "retained_frame_start": 100, "seconds_per_frame": .01,
                "probability_path": str(tail), "segments": segment_retained_probabilities(values, first_frame=0),
                "full_probability_archive": metadata}, raw

    def test_continuous_export_retains_prefix_after_compaction_and_resegments_whole(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            data, raw = self.fixture(root)
            self.assertEqual(Path(data["full_probability_archive"]["path"]).read_bytes(), raw)
            self.assertFalse((root / "whole.f32.partial").exists())
            result = verify_full_probability_archive(data, root)
            self.assertEqual(result["full_native_frames_verified"], 201)
            self.assertEqual(result["native_default_segments_compared"], 2)
            self.assertEqual(data["full_probability_archive"]["overlap_checks"], 2)

    def test_native_rewrite_is_rejected_before_appending_any_new_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = ProbabilityArchive(Path(tmp) / "whole.f32")
            archive.update(first_frame=0, frame_count=10, retained_bytes=array.array("f", [.3]*80).tobytes())
            changed = array.array("f", [.3] * 80)
            changed[0] = .4
            with self.assertRaisesRegex(ValueError, "rewrote"):
                archive.update(first_frame=5, frame_count=15, retained_bytes=changed.tobytes())
            self.assertEqual(archive.frames, 10)
            archive.close()
            self.assertEqual(archive.partial.stat().st_size, 10*32)
            self.assertFalse(archive.path.exists())

    def test_missing_prefix_regression_and_invalid_probability_never_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = ProbabilityArchive(Path(tmp) / "whole.f32")
            archive.update(first_frame=0, frame_count=10, retained_bytes=array.array("f", [.3]*80).tobytes())
            for first, count, raw in [(11, 12, b"\0"*32), (0, 9, b"\0"*9*32), (0, 11, b"short")]:
                with self.subTest(first=first, count=count), self.assertRaises(ValueError):
                    archive.update(first_frame=first, frame_count=count, retained_bytes=raw)
            for bad in (float("nan"), float("inf"), -1., 1.1):
                raw = array.array("f", [.3]*80 + [bad]*8).tobytes()
                with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "invalid native probability"):
                    archive.update(first_frame=0, frame_count=11, retained_bytes=raw)
            self.assertEqual(archive.frames, 10)
            archive.close()
            self.assertFalse(archive.path.exists())

    def test_interrupted_archive_is_partial_and_wrong_final_count_does_not_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = ProbabilityArchive(Path(tmp) / "whole.f32")
            archive.update(first_frame=0, frame_count=1, retained_bytes=array.array("f", [.2]*8).tobytes())
            with self.assertRaisesRegex(ValueError, "incomplete"):
                archive.finish(frame_count=2)
            archive.close()
            self.assertTrue(archive.partial.is_file())
            self.assertFalse(archive.path.exists())

    def test_completed_or_racing_destination_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = ProbabilityArchive(root / "whole.f32")
            archive.update(first_frame=0, frame_count=1, retained_bytes=array.array("f", [.2]*8).tobytes())
            archive.path.write_bytes(b"other completed evidence")
            with self.assertRaises(FileExistsError):
                archive.finish(frame_count=1)
            archive.close()
            self.assertEqual(archive.path.read_bytes(), b"other completed evidence")
            with self.assertRaises(ValueError):
                ProbabilityArchive(archive.path)

    def test_changed_hash_or_scope_and_unverified_preview_contract_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            data, _ = self.fixture(root)
            for change in ({"sha256": "0"*64}, {"stream_contract": "preview-allowed"},
                           {"completed": False}, {"first_frame": 1}, {"frame_count": 100}):
                bad = copy.deepcopy(data)
                bad["full_probability_archive"].update(change)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    verify_full_probability_archive(bad, root)

    def test_final_tail_mismatch_is_rejected_even_with_valid_full_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            data, _ = self.fixture(root)
            Path(data["probability_path"]).write_bytes(b"\0"*101*32)
            with self.assertRaisesRegex(ValueError, "tail disagree"):
                verify_full_probability_archive(data, root)

    def test_full_frames_must_reproduce_native_frozen_prefix_not_just_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            data, raw = self.fixture(root)
            path = Path(data["full_probability_archive"]["path"])
            path.write_bytes(b"\0"*100*32 + raw[100*32:])
            data["full_probability_archive"]["sha256"] = file_hash(path)
            with self.assertRaisesRegex(ValueError, "reproduce final native segments"):
                verify_full_probability_archive(data, root)


if __name__ == "__main__":
    unittest.main()
