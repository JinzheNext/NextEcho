"""Native-result/identity/resource contracts; synthetic files, no C library load."""
from __future__ import annotations

import array
import copy
import ctypes
import json
import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from workbench.contracts import PipelineError
from workbench.io_utils import file_hash, write_json
from workbench.nemotron_candidate import probe_nemotron, validate_candidate
from workbench.nemotron_worker import ModelConfig, NativeSegment, SegmentationConfig


class NemotronCandidateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.environment = patch.dict(os.environ, {"NEXTECHO_RUNTIME_DIR": str(self.root / "runtime"), "NEXTECHO_JOB_ID": ""})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.tmp.cleanup()

    def native(self, folder):
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "native.probabilities.f32"
        path.write_bytes(array.array("f", [0.25] * 101 * 8).tobytes())
        return {"schema_version": 1, "artifact_kind": "nemotron-native-candidate", "frame_count": 101,
            "speaker_slot_capacity": 8, "seconds_per_frame": 0.01, "retained_frame_start": 0,
            "origin_ms": 79178, "duration_seconds": 1, "backend_requested": "metal", "preset": "v3-offline",
            "full_attention": False, "probability_path": str(path), "probability_sha256": file_hash(path),
            "segments": [{"start": 0, "end": 0.5, "speaker": 1}, {"start": 0.25, "end": 1, "speaker": 2}]}

    def assets(self):
        model = self.root / "toy.gguf"
        model.write_bytes(b"not a real model; identity boundary mocked")
        runtime = self.root / "assets"
        lib = runtime / "package/lib/libnemo_speech_asr_c.1.dylib"
        lib.parent.mkdir(parents=True)
        lib.write_bytes(b"no executable code; never loaded")
        audio = self.root / "clip.wav"
        with wave.open(str(audio), "wb") as f:
            f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            f.writeframes(b"\0\0" * 16000)
        digest = "08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1"
        ledger = self.root / "ledger.json"
        write_json(ledger, {"model": {"path": str(model), "revision": "f667ed73aee57d40cc39428eb768b4fd87a0a29e", "sha256": digest},
            "code": {"commit": "6a3ca369370782acee0dd155e34394ba60b920c4", "release_archive_sha256": "5cb02ba7c04f0b5585ce5cde9c830be5f0b7dfb4c83083c500109c381b4f2da9"},
            "runtime_files": {str(lib.relative_to(runtime)): {"sha256": file_hash(lib)}}, "runtime_symlinks": {}})
        def identity(path):
            return digest if Path(path) == model else file_hash(Path(path))
        return audio, runtime, ledger, identity

    def test_native_overlap_and_10ms_padding_preserved_without_word_times_or_names(self):
        data = self.native(self.root / "probe")
        self.assertIs(validate_candidate(data, self.root / "probe", duration=1, origin_ms=79178), data)
        self.assertEqual(data["segments"][1]["start"], 0.25)
        self.assertEqual(data["frame_count"], 101)
        self.assertNotIn("words", data)
        self.assertNotIn("name", data)

    def test_bad_intervals_channels_origin_cadence_and_compaction_are_blocked(self):
        valid = self.native(self.root / "probe")
        changes = [{"origin_ms": 0}, {"retained_frame_start": 1}, {"frame_count": True}, {"seconds_per_frame": 0.08},
                   {"speaker_slot_capacity": 4}, {"preset": "v3-streaming"}, {"full_attention": True},
                   {"segments": [{"start": 0, "end": float("nan"), "speaker": 1}]},
                   {"segments": [{"start": -1, "end": 1, "speaker": 1}]},
                   {"segments": [{"start": 0.2, "end": 0.1, "speaker": 1}]},
                   {"segments": [{"start": 0, "end": 1, "speaker": 9}]}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(PipelineError):
                validate_candidate({**valid, **change}, self.root / "probe", duration=1, origin_ms=79178)

    def test_probabilities_require_scope_exact_bytes_hash_and_real_probability_values(self):
        folder = self.root / "probe"
        data = self.native(folder)
        path = Path(data["probability_path"])
        for value in (float("nan"), float("inf"), -0.01, 1.01):
            path.write_bytes(array.array("f", [value] * 101 * 8).tobytes())
            data["probability_sha256"] = file_hash(path)
            with self.assertRaises(PipelineError):
                validate_candidate(data, folder, duration=1, origin_ms=79178)
        path.write_bytes(b"partial")
        with self.assertRaises(PipelineError):
            validate_candidate(data, folder, duration=1, origin_ms=79178)
        outside = self.root / "outside.f32"
        outside.write_bytes(b"x")
        with self.assertRaises(PipelineError):
            validate_candidate({**data, "probability_path": str(outside)}, folder, duration=1, origin_ms=79178)

    def test_only_short_pcm16_probes_reach_weight_or_library_checks(self):
        audio = self.root / "long.wav"
        with wave.open(str(audio), "wb") as f:
            f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            f.writeframes(b"\0\0" * 16000 * 301)
        with patch("workbench.nemotron_candidate.run_owned") as run:
            with self.assertRaises(PipelineError) as error:
                probe_nemotron(audio, self.root / "probe", ledger=self.root / "not-read", runtime_root=self.root, origin_ms=0)
            self.assertEqual(error.exception.code, "INVALID_INPUT")
            run.assert_not_called()

    def test_fake_owned_boundary_gets_resource_floor_and_candidate_never_promotes(self):
        audio, runtime, ledger, identity = self.assets()
        folder = self.root / "probe"
        def owned(command, **kwargs):
            self.assertGreaterEqual(kwargs["policy"].reserve_bytes, 3 * 1024**3)
            self.assertEqual(kwargs["policy"].max_rss_bytes, 4 * 1024**3)
            self.assertLessEqual(kwargs["policy"].timeout_seconds, 600)
            self.assertEqual(command[0], "/usr/bin/env")
            write_json(folder / "native.json", self.native(folder))
            return {"elapsed_seconds": 1, "peak_worker_rss_bytes": 1}
        with patch("workbench.nemotron_candidate.file_hash", side_effect=identity), patch("workbench.nemotron_candidate.run_owned", side_effect=owned):
            result = probe_nemotron(audio, folder, ledger=ledger, runtime_root=runtime, origin_ms=79178)
        self.assertFalse(result["default_promotion"])
        self.assertFalse(result["precise_alignment_release"])
        self.assertEqual(result["quality_validation"], "pending-author-gold")

    def test_cancelled_probe_has_failure_evidence_and_no_native_success(self):
        audio, runtime, ledger, identity = self.assets()
        folder = self.root / "probe"
        with patch("workbench.nemotron_candidate.file_hash", side_effect=identity), patch("workbench.nemotron_candidate.run_owned") as run:
            with self.assertRaises(PipelineError) as error:
                probe_nemotron(audio, folder, ledger=ledger, runtime_root=runtime, origin_ms=79178, cancel=lambda: True)
            self.assertEqual(error.exception.code, "CANCELLED")
            run.assert_not_called()
        self.assertFalse((folder / "native.json").exists())
        self.assertEqual(json.loads((folder / "probe.failure.json").read_text())["error_code"], "CANCELLED")

    def test_changed_runtime_link_or_digest_blocks_before_owned_worker(self):
        audio, runtime, ledger, identity = self.assets()
        original = json.loads(ledger.read_text())
        for change in ("digest", "link"):
            data = copy.deepcopy(original)
            if change == "digest":
                data["code"]["release_archive_sha256"] = "wrong-release"
            else:
                path = runtime / "package/lib/alias.dylib"
                path.symlink_to(self.root / "outside")
                data["runtime_symlinks"][str(path.relative_to(runtime))] = "../../outside"
            write_json(ledger, data)
            with patch("workbench.nemotron_candidate.file_hash", side_effect=identity), patch("workbench.nemotron_candidate.run_owned") as run:
                with self.assertRaises(PipelineError):
                    probe_nemotron(audio, self.root / "probe", ledger=ledger, runtime_root=runtime, origin_ms=79178)
                run.assert_not_called()

    def test_explicit_long_mode_uses_tail_protocol_and_bounded_budget(self):
        audio, runtime, ledger, identity = self.assets()
        folder = self.root / "long-explicit"
        def owned(command, **kwargs):
            self.assertEqual(kwargs["policy"].timeout_seconds, 900)
            request = json.loads(Path(command[-1]).read_text())
            self.assertTrue(request["long_context"])
            self.assertEqual(file_hash(folder / "nemotron_worker.py"), request["worker_sha256"])
            self.assertEqual(file_hash(folder / "nemotron_timeline.py"), request["timeline_sha256"])
            native = self.native(folder)
            native.update(schema_version=2, segments=[], timeline_protocol="native-retained-tail-v2",
                probability_scope="retained-tail", segments_scope="native-whole-timeline", completed=True,
                input_samples_consumed=16000, resume_granularity="whole-stage")
            checkpoint = folder / "final.json"
            write_json(checkpoint, {"frame_count": 101, "retained_frame_start": 0,
                                   "input_samples_consumed": 16000, "segments": []})
            native["checkpoints"] = [{"path": str(checkpoint), "sha256": file_hash(checkpoint)}]
            write_json(folder / "native.json", native)
            return {}
        with patch("workbench.nemotron_candidate.file_hash", side_effect=identity), patch("workbench.nemotron_candidate.run_owned", side_effect=owned):
            result = probe_nemotron(audio, folder, ledger=ledger, runtime_root=runtime,
                                    origin_ms=79178, long_context=True)
        self.assertTrue(result["long_context"])
        self.assertFalse(result["default_promotion"])

    def test_restart_replays_same_input_in_new_directory_and_rejects_completed_or_changed(self):
        audio, runtime, ledger, identity = self.assets()
        previous, retry = self.root / "cancelled", self.root / "retry"
        with patch("workbench.nemotron_candidate.file_hash", side_effect=identity):
            with self.assertRaises(PipelineError):
                probe_nemotron(audio, previous, ledger=ledger, runtime_root=runtime, origin_ms=79178, cancel=lambda: True)
            original_request = (previous / "request.json").read_bytes()
            def owned(command, **kwargs):
                request = json.loads(Path(command[-1]).read_text())
                self.assertEqual(request["recovery"]["mode"], "whole-stage-replay-no-state-import")
                self.assertEqual(request["recovery"]["parent_request_sha256"], file_hash(previous / "request.json"))
                write_json(retry / "native.json", self.native(retry))
                return {}
            with patch("workbench.nemotron_candidate.run_owned", side_effect=owned):
                probe_nemotron(audio, retry, ledger=ledger, runtime_root=runtime, origin_ms=79178, restart_from=previous)
            self.assertEqual((previous / "request.json").read_bytes(), original_request)
            with patch("workbench.nemotron_candidate.run_owned") as run:
                for folder, parent, origin in [("already-complete", retry, 79178), ("changed-origin", previous, 79179)]:
                    with self.subTest(folder=folder), self.assertRaises(PipelineError) as error:
                        probe_nemotron(audio, self.root / folder, ledger=ledger, runtime_root=runtime, origin_ms=origin, restart_from=parent)
                    self.assertEqual(error.exception.code, "INVALID_INPUT")
                run.assert_not_called()

    def test_restart_never_enters_worker_while_previous_owned_group_is_live(self):
        audio, runtime, ledger, identity = self.assets()
        previous = self.root / "previous"
        with patch("workbench.nemotron_candidate.file_hash", side_effect=identity):
            with self.assertRaises(PipelineError):
                probe_nemotron(audio, previous, ledger=ledger, runtime_root=runtime, origin_ms=79178, cancel=lambda: True)
            write_json(previous / "worker.guard.owner.json", {"pid": 999999, "pgid": 999999, "birth": "synthetic"})
            with patch("workbench.resources.process_alive", return_value=True), patch("workbench.nemotron_candidate.run_owned") as run:
                with self.assertRaises(PipelineError):
                    probe_nemotron(audio, self.root / "new", ledger=ledger, runtime_root=runtime, origin_ms=79178, restart_from=previous)
                run.assert_not_called()

    def test_restart_never_creates_nested_output_or_accepts_changed_controller(self):
        audio, runtime, ledger, identity = self.assets()
        previous = self.root / "previous"
        with patch("workbench.nemotron_candidate.file_hash", side_effect=identity):
            with self.assertRaises(PipelineError):
                probe_nemotron(audio, previous, ledger=ledger, runtime_root=runtime, origin_ms=79178, cancel=lambda: True)
            nested = previous / "new-attempt"
            with patch("workbench.nemotron_candidate.run_owned") as run:
                with self.assertRaises(PipelineError):
                    probe_nemotron(audio, nested, ledger=ledger, runtime_root=runtime, origin_ms=79178, restart_from=previous)
                self.assertFalse(nested.exists())
                request = json.loads((previous / "request.json").read_text())
                request["controller_sha256"] = "different-implementation"
                write_json(previous / "request.json", request)
                with self.assertRaises(PipelineError):
                    probe_nemotron(audio, self.root / "changed", ledger=ledger, runtime_root=runtime, origin_ms=79178, restart_from=previous)
                run.assert_not_called()

    def test_full_probability_archive_is_explicit_long_mode_only_before_any_audio_read(self):
        with patch("workbench.nemotron_candidate.run_owned") as run:
            for options in ({"retain_full_probabilities": True}, {"long_context": True, "retain_full_probabilities": 1}):
                with self.subTest(options=options), self.assertRaises(PipelineError) as error:
                    probe_nemotron(self.root / "missing.wav", self.root / "out", ledger=self.root / "missing-ledger",
                                    runtime_root=self.root, origin_ms=0, **options)
                self.assertEqual(error.exception.code, "INVALID_INPUT")
            run.assert_not_called()

    def test_requested_full_archive_is_verified_and_missing_archive_reports_cause(self):
        from workbench.nemotron_probabilities import ProbabilityArchive
        audio, runtime, ledger, identity = self.assets()
        def owned(command, **kwargs):
            request = json.loads(Path(command[-1]).read_text())
            folder = Path(command[-1]).parent
            self.assertTrue(request["retain_full_probabilities"])
            self.assertEqual(file_hash(folder / "nemotron_probabilities.py"), request["probabilities_code_sha256"])
            native = self.native(folder)
            native.update(schema_version=2, segments=[], timeline_protocol="native-retained-tail-v2",
                probability_scope="retained-tail", segments_scope="native-whole-timeline", completed=True,
                input_samples_consumed=16000, resume_granularity="whole-stage")
            checkpoint = folder / "final.json"
            write_json(checkpoint, {"frame_count": 101, "retained_frame_start": 0,
                                   "input_samples_consumed": 16000, "segments": []})
            native["checkpoints"] = [{"path": str(checkpoint), "sha256": file_hash(checkpoint)}]
            if folder.name != "missing-archive":
                archive = ProbabilityArchive(folder / "whole.f32")
                archive.update(first_frame=0, frame_count=101, retained_bytes=Path(native["probability_path"]).read_bytes())
                native["full_probability_archive"] = archive.finish(frame_count=101)
            write_json(folder / "native.json", native)
            return {}
        with patch("workbench.nemotron_candidate.file_hash", side_effect=identity), patch("workbench.nemotron_candidate.run_owned", side_effect=owned):
            result = probe_nemotron(audio, self.root / "full", ledger=ledger, runtime_root=runtime,
                                    origin_ms=79178, long_context=True, retain_full_probabilities=True)
            self.assertTrue(result["full_probability_archive_validated"])
            with self.assertRaises(PipelineError):
                probe_nemotron(audio, self.root / "missing-archive", ledger=ledger, runtime_root=runtime,
                                origin_ms=79178, long_context=True, retain_full_probabilities=True)
            failure = json.loads((self.root / "missing-archive/probe.failure.json").read_text())
            self.assertEqual(failure["cause_type"], "ValueError")
            self.assertIn("archive is missing", failure["cause_message"])

    @unittest.skipUnless(ctypes.sizeof(ctypes.c_void_p) == 8, "64-bit pinned ABI")
    def test_pinned_c_abi_struct_layouts_include_geometry_and_segment_padding(self):
        self.assertEqual((ctypes.sizeof(ModelConfig), ctypes.sizeof(SegmentationConfig), ctypes.sizeof(NativeSegment)), (56, 48, 24))
        self.assertEqual((ModelConfig.gpu.offset, ModelConfig.preset.offset, ModelConfig.left_context_frames.offset), (16, 24, 40))


if __name__ == "__main__":
    unittest.main()
