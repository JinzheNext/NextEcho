"""Synthetic model boundaries, real shared routing/caches/writers/identity guards."""
from __future__ import annotations

import array
import contextlib
import json
import os
import shutil
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

from workbench.config import NextEchoConfig
from workbench.contracts import PipelineError
from workbench.io_utils import file_hash, write_json
from workbench.nemotron_probabilities import ProbabilityArchive
from workbench.nemotron_speaker import BACKEND, UNCERTAIN_BOUNDARY, binding, posterior_cores, priority_activity
from workbench.nemotron_timeline import segment_retained_probabilities
from workbench.speaker_service import build_item_speakers, speaker_artifact_paths
from workbench.speaker_artifacts import read_speaker_file


class NativeSpeakerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.folder = self.root / "item"
        self.folder.mkdir()
        self.audio = self.folder / "audio.wav"
        with wave.open(str(self.audio), "wb") as f:
            f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            f.writeframes(b"\0" * 32000 * 2)
        self.raw = {"transcription": [{"id": "r1", "text": "甲乙丙丁戊", "offsets": {"from": 0, "to": 2000}}]}
        write_json(self.folder / "transcript.json", self.raw)
        self.item = {"source_id": "fixture-recording", "source_label": "item", "audio_path": str(self.audio),
                     "asr": {"actual_engine": "sensevoice", "timing_precision": "vad-segment"}}
        write_json(self.folder / "metadata.json", self.item)
        ctc = self.root / "ctc"
        ctc.mkdir()
        for name in ("model.int8.onnx", "tokens.txt"):
            (ctc / name).write_text("synthetic weights")
        self.ledger, self.library = self.root / "ledger.json", self.root / "lib.dylib"
        self.ledger.write_text("{}")
        self.library.write_bytes(b"synthetic native library")
        self.locked = {"model": {"path": str(self.library), "sha256": file_hash(self.library), "revision": "fixture-revision"}}
        self.settings = {"backend": BACKEND, "worker_python": sys.executable, "ctc_model_path": str(ctc),
                         "nemotron": {"asset_ledger": str(self.ledger), "runtime_root": str(self.root),
                                      "attribution_profile": "native-padded"}}
        self.native_calls, self.ctc_calls = [], []

    def fake_native(self, audio, folder, **kwargs):
        self.native_calls.append((folder, kwargs))
        folder.mkdir()
        code_root = Path(__file__).parents[1] / "workbench"
        request = {"audio_sha256": file_hash(audio), "model_sha256": self.locked["model"]["sha256"],
                   "audio_path": str(audio.resolve()), "output": str(folder / "native.json"), "progress": str(folder / "worker.state.json"),
                   "ledger_sha256": file_hash(self.ledger), "library": str(self.library), "origin_ms": 0,
                   "long_context": True, "retain_full_probabilities": True}
        for key, name in [("worker_sha256", "nemotron_worker.py"), ("timeline_sha256", "nemotron_timeline.py"),
                          ("probabilities_code_sha256", "nemotron_probabilities.py"), ("controller_sha256", "nemotron_candidate.py")]:
            request[key] = file_hash(code_root / name)
        write_json(folder / "request.json", request)
        values = array.array("f", [0.] * 201 * 8)
        for frame in range(95):
            values[frame * 8] = .9
        for frame in range(100, 190):
            values[frame * 8 + 1] = .9
        archive = ProbabilityArchive(folder / "whole.f32")
        archive.update(first_frame=0, frame_count=201, retained_bytes=values.tobytes())
        full = archive.finish(frame_count=201)
        tail = folder / "tail.f32"
        tail.write_bytes(values.tobytes())
        segments = segment_retained_probabilities(values, first_frame=0)
        checkpoint = folder / "checkpoint.json"
        write_json(checkpoint, {"frame_count": 201, "retained_frame_start": 0,
                               "input_samples_consumed": 32000, "segments": segments})
        native = {"schema_version": 2, "artifact_kind": "nemotron-native-candidate", "frame_count": 201,
                  "speaker_slot_capacity": 8, "seconds_per_frame": .01, "retained_frame_start": 0,
                  "origin_ms": 0, "duration_seconds": 2., "backend_requested": "metal", "preset": "v3-offline",
                  "full_attention": False, "probability_path": str(tail), "probability_sha256": file_hash(tail),
                  "segments": segments, "timeline_protocol": "native-retained-tail-v2", "completed": True,
                  "input_samples_consumed": 32000, "probability_scope": "retained-tail",
                  "segments_scope": "native-whole-timeline", "resume_granularity": "whole-stage",
                  "checkpoints": [{"path": str(checkpoint), "sha256": file_hash(checkpoint)}], "full_probability_archive": full}
        write_json(folder / "native.json", native)
        write_json(folder / "probe.result.json", {"status": "candidate-probe-complete", "native": native, "resources": {"synthetic": True}})

    def fake_ctc(self, command, **kwargs):
        request = json.loads(Path(command[-1]).read_text())
        self.assertEqual(request["operation"], "ctc")
        self.ctc_calls.append(request)
        tokens = [{"text": c, "start_ms": t, "end_ms": None} for c, t in zip("甲乙丙丁戊", [200, 800, 970, 1200, 1800])]
        write_json(Path(request["output"]), {"schema_version": 1, "clips": [{"id": "full", "tokens": tokens}]})
        return {"synthetic": True}

    @contextlib.contextmanager
    def synthetic_models(self):
        with contextlib.ExitStack() as stack:
            for name, kwargs in [
                ("workbench.nemotron_speaker.platform.system", {"return_value": "Darwin"}),
                ("workbench.nemotron_speaker.platform.machine", {"return_value": "arm64"}),
                ("workbench.nemotron_speaker.validate_assets", {"return_value": (self.locked, self.library)}),
                ("workbench.nemotron_speaker.probe_nemotron", {"side_effect": self.fake_native}),
                ("workbench.acoustic_dialogue.run_media_command", {"return_value": SimpleNamespace(returncode=0, stdout='{"python":"fixture","packages":{}}')}),
                ("workbench.resources.run_owned", {"side_effect": self.fake_ctc}),
                ("workbench.resources.model_policy", {"return_value": object()}),
                ("workbench.speaker_service._build_legacy_speakers", {"side_effect": AssertionError("no fallback")}),
            ]:
                stack.enter_context(patch(name, **kwargs))
            stack.enter_context(patch.dict(os.environ, {"NEXTECHO_RUNTIME_DIR": str(self.root / "runtime")}))
            yield

    def build(self):
        return build_item_speakers(self.folder, self.item, config=NextEchoConfig({"diarization": self.settings}))

    def test_shared_route_exports_voices_evidence_and_partial_without_names_or_text_loss(self):
        with self.synthetic_models():
            original = file_hash(self.folder / "transcript.json")
            result = self.build()
            self.assertEqual(file_hash(self.folder / "transcript.json"), original)
            payload = result["payload"]
            self.assertEqual(payload["backend"], BACKEND)
            self.assertEqual(payload["release_status"], "candidate")
            self.assertEqual(payload["recording_identity"]["status"], "verified-recording-stage")
            self.assertEqual(len(payload["speakers"]), 2)
            self.assertTrue(all(p["identity_status"] == "unknown" for p in payload["speakers"]))
            self.assertEqual("".join(r["text"] for r in payload["speaker_turns"]), "甲乙丙丁戊")
            self.assertEqual(self.item["speaker_status"], "partial")
            self.assertTrue(Path(self.item["diarization_raw_json_path"]).is_relative_to(self.folder / "speaker-stages"))
            self.assertTrue(Path(self.item["native_probability_path"]).is_file())
            self.assertFalse(payload["scoring_eligible"])
            self.assertEqual(len(self.native_calls), 1)
            self.assertEqual(len(self.ctc_calls), 1)

    def test_profile_and_text_changes_reuse_audio_stages_but_keep_ids_and_uncertainty(self):
        with self.synthetic_models():
            first = self.build()["payload"]
            self.settings["nemotron"]["attribution_profile"] = "posterior-priority"
            second = self.build()["payload"]
            self.assertEqual(len(self.native_calls), 1)
            self.assertEqual(len(self.ctc_calls), 1)
            self.assertTrue(second["resources"][BACKEND]["cached"])
            self.assertEqual(first["recording_identity"], second["recording_identity"])
            self.assertEqual([p["speaker_uuid"] for p in first["speakers"]], [p["speaker_uuid"] for p in second["speakers"]])
            self.assertEqual([r["speaker"] for r in second["speaker_turns"]], ["说话人 1", "归属待核对", "说话人 2"])
            self.assertNotIn(UNCERTAIN_BOUNDARY, [p["raw_labels"][0] for p in second["speakers"]])
            self.raw["transcription"][0]["text"] += "。"
            write_json(self.folder / "transcript.json", self.raw)
            third = self.build()["payload"]
            self.assertEqual("".join(r["text"] for r in third["speaker_turns"]), "甲乙丙丁戊。")
            self.assertEqual(len(self.native_calls), 1)
            self.assertEqual(len(self.ctc_calls), 1)

    def test_tampered_probability_stops_without_new_inference_or_legacy_fallback(self):
        with self.synthetic_models():
            first = self.build()
            path = Path(first["artifact_paths"]["native_probability_path"])
            path.write_bytes(b"\0" * path.stat().st_size)
            with self.assertRaises(PipelineError) as caught:
                self.build()
            self.assertEqual(caught.exception.code, "CACHE_MISMATCH")
            self.assertEqual(len(self.native_calls), 1)

    def test_interrupted_parent_recovers_completed_native_before_retrying_failed_ctc(self):
        with self.synthetic_models():
            with patch("workbench.resources.run_owned", side_effect=PipelineError("RESOURCE_LIMIT", "fixture CTC stop")):
                with self.assertRaises(PipelineError) as caught:
                    self.build()
            self.assertEqual(caught.exception.code, "RESOURCE_LIMIT")
            pointer = self.folder / "speaker-stages/nemotron/attempt.json"
            record = json.loads(pointer.read_text())
            attempt = pointer.parent / record["directory"]
            original = {p: file_hash(p) for p in attempt.iterdir() if p.is_file()}
            # Simulate parent death after probe/result but before evidence hash
            # pointer publication; a complete verified probe is not rerun.
            record.pop("hashes")
            write_json(pointer, record)
            result = self.build()
            self.assertEqual(len(self.native_calls), 1)
            self.assertEqual(len(self.ctc_calls), 1)
            self.assertTrue(result["payload"]["resources"][BACKEND]["cached"])
            self.assertTrue(all(file_hash(p) == digest for p, digest in original.items()))

    def test_cancelled_native_is_replayed_in_new_attempt_without_overwriting_old_evidence(self):
        with self.synthetic_models():
            def cancelled(audio, folder, **kwargs):
                self.fake_native(audio, folder, **kwargs)
                (folder / "probe.result.json").unlink()
                write_json(folder / "probe.failure.json", {"error_code": "CANCELLED"})
                raise PipelineError("CANCELLED", "fixture stop")
            with patch("workbench.nemotron_speaker.probe_nemotron", side_effect=cancelled):
                with self.assertRaises(PipelineError) as caught:
                    self.build()
            self.assertEqual(caught.exception.code, "CANCELLED")
            old = self.native_calls[0][0]
            original = {p: file_hash(p) for p in old.iterdir() if p.is_file()}
            result = self.build()
            self.assertEqual(len(self.native_calls), 2)
            self.assertEqual(self.native_calls[-1][1]["restart_from"], old)
            self.assertNotEqual(self.native_calls[-1][0], old)
            self.assertTrue(all(file_hash(p) == digest for p, digest in original.items()))
            self.assertEqual(result["payload"]["acoustic_audit"]["resume_granularity"], "whole-stage")

    def test_old_voice_linking_is_retained_on_disk_but_not_registered_as_native_evidence(self):
        stage = self.folder / "speaker-stages"
        stage.mkdir()
        old = stage / "voice-linking.json"
        old.write_text("old backend evidence")
        self.item["voice_linking_json_path"] = str(old)
        with self.synthetic_models():
            result = self.build()
            self.assertNotIn("voice_linking_json_path", result["artifact_paths"])
            self.assertNotIn("voice_linking_json_path", self.item)
            self.assertEqual(old.read_text(), "old backend evidence")

    def test_portable_binding_rejects_missing_relative_and_invalid_profile_before_models(self):
        for row in ({}, {"asset_ledger": "relative.json", "runtime_root": str(self.root)},
                    {"asset_ledger": str(self.ledger), "runtime_root": str(self.root), "attribution_profile": "force-two"}):
            with self.subTest(row=row), self.assertRaises(PipelineError) as caught:
                binding({"nemotron": row})
            self.assertEqual(caught.exception.code, "CONFIG_ERROR")

    def test_private_evidence_cannot_escape_item_or_change_after_export(self):
        with self.synthetic_models():
            payload = self.build()["payload"]
            key = "native_probability_path"
            payload["acoustic_evidence"]["paths"][key] = str(self.library)
            payload["acoustic_evidence"]["hashes"][key] = file_hash(self.library)
            with self.assertRaises(PipelineError) as caught:
                speaker_artifact_paths(self.folder, payload=payload)
            self.assertEqual(caught.exception.code, "CACHE_MISMATCH")

    def test_export_cannot_replace_canonical_native_evidence_with_another_valid_local_file(self):
        with self.synthetic_models():
            self.build()
            path = self.folder / "transcript.speakers.json"
            export = json.loads(path.read_text())
            export["acoustic_evidence"]["paths"]["native_probability_path"] = str(self.audio)
            export["acoustic_evidence"]["hashes"]["native_probability_path"] = file_hash(self.audio)
            write_json(path, export)
            with self.assertRaises(PipelineError) as caught:
                read_speaker_file(path)
            self.assertEqual(caught.exception.code, "CACHE_MISMATCH")

    def test_diagnostics_use_native_assets_and_ctc_without_requiring_sherpa_diarizer_weights(self):
        from workbench.capabilities import acoustic_speaker_status
        from workbench.speaker_transcript import detect_speaker_backend_status
        with self.synthetic_models(), patch("workbench.capabilities.probe_environment", return_value={"available": True}):
            config = NextEchoConfig({"diarization": self.settings})
            status = acoustic_speaker_status(config)
            self.assertEqual(status["status"], "ready")
            self.assertEqual(status["release_status"], "candidate")
            legacy_shape = detect_speaker_backend_status(config)
            self.assertTrue(legacy_shape["ready"])
            self.assertEqual(legacy_shape["selected_backend"], BACKEND)
            self.assertEqual(self.native_calls, [])
            self.assertEqual(self.ctc_calls, [])

    def resume_copy(self):
        child_run = self.root / "resumed"
        child = child_run / "item"
        shutil.copytree(self.folder, child)
        item = {**self.item, "audio_path": str(child / "audio.wav")}
        write_json(child / "metadata.json", item)
        config = NextEchoConfig({"diarization": self.settings,
            "runtime": {"resume_verified": True, "resume_parent_dir": str(self.root), "resume_run_dir": str(child_run)}})
        return child, item, config

    def test_verified_resume_copy_relocates_only_a_new_view_and_reuses_both_models(self):
        with self.synthetic_models():
            first = self.build()["payload"]
            original = {p: file_hash(p) for p in self.folder.rglob("*") if p.is_file()}
            child, item, config = self.resume_copy()
            copied = child / Path(item["native_candidate_json_path"]).relative_to(self.folder)
            copied_digest = file_hash(copied)
            result = build_item_speakers(child, item, config=config)["payload"]
            self.assertEqual(len(self.native_calls), 1)
            self.assertEqual(len(self.ctc_calls), 1)
            self.assertEqual(result["recording_identity"], first["recording_identity"])
            self.assertTrue(result["resources"][BACKEND]["cached"])
            self.assertTrue(result["resources"]["ctc"]["cached"])
            self.assertTrue(all(file_hash(p) == digest for p, digest in original.items()))
            native_path = Path(item["native_candidate_json_path"])
            self.assertNotEqual(native_path, copied)
            self.assertEqual(file_hash(copied), copied_digest)
            native = json.loads(native_path.read_text())
            relocation = native["relocations"][-1]
            self.assertEqual(relocation["kind"], "verified-copy-path-view-not-new-inference")
            for source in relocation["originals"].values():
                self.assertEqual(file_hash(Path(source["path"])), source["sha256"])
            self.assertTrue(Path(native["probability_path"]).is_relative_to(child))
            self.assertTrue(all(Path(p).is_relative_to(child) for p in result["acoustic_evidence"]["paths"].values()))

    def test_damaged_resume_copy_is_not_repaired_into_a_verified_model_result(self):
        with self.synthetic_models():
            self.build()
            child, item, config = self.resume_copy()
            data = child / Path(item["native_probability_path"]).relative_to(self.folder)
            data.write_bytes(b"\0" * data.stat().st_size)
            with self.assertRaises(PipelineError) as caught:
                build_item_speakers(child, item, config=config)
            self.assertEqual(caught.exception.code, "CACHE_MISMATCH")
            self.assertEqual(len(self.native_calls), 1)

    def test_resume_through_blocked_intermediate_reuses_verified_grandparent_copy(self):
        with self.synthetic_models():
            self.build()
            child, _, _ = self.resume_copy()  # interrupted before native relocation
            original = {p: file_hash(p) for p in self.folder.rglob('*') if p.is_file()}
            next_run = self.root / 'twice'
            target = next_run / 'item'
            shutil.copytree(child, target)
            item = {**self.item, 'audio_path': str(target / 'audio.wav')}
            config = NextEchoConfig({'diarization': self.settings, 'runtime': {
                'resume_verified': True, 'resume_parent_dir': str(child.parent), 'resume_run_dir': str(next_run)}})
            build_item_speakers(target, item, config=config)
            self.assertEqual(len(self.native_calls), 1)
            self.assertEqual(len(self.ctc_calls), 1)
            native = json.loads(Path(item['native_candidate_json_path']).read_text())
            self.assertTrue(Path(native['probability_path']).is_relative_to(target))
            self.assertTrue(all(file_hash(p) == digest for p, digest in original.items()))

    def test_interrupted_relocation_preserves_copied_original_and_retries_without_inference(self):
        from workbench import nemotron_speaker as module
        with self.synthetic_models():
            self.build()
            child, item, config = self.resume_copy()
            pointer = child / "speaker-stages/nemotron/attempt.json"
            pointer_bytes = pointer.read_bytes()
            copied = pointer.parent / json.loads(pointer_bytes)["directory"]
            original = {p: file_hash(p) for p in copied.rglob("*") if p.is_file()}
            real_write = module.write_json
            def interrupted(path, value):
                if path.name == "native.json":
                    raise OSError("simulated publication interruption")
                return real_write(path, value)
            with patch.object(module, "write_json", side_effect=interrupted):
                with self.assertRaises(PipelineError):
                    build_item_speakers(child, item, config=config)
            self.assertEqual(pointer.read_bytes(), pointer_bytes)
            self.assertTrue(all(file_hash(p) == digest for p, digest in original.items()))
            result = build_item_speakers(child, item, config=config)
            self.assertEqual(result["payload"]["backend"], BACKEND)
            self.assertEqual(len(self.native_calls), 1)
            self.assertEqual(len(self.ctc_calls), 1)

    def test_priority_preserves_real_multi_core_and_padded_only_uncertainty(self):
        padded = [{"start": 0., "end": .5, "speaker": 1}, {"start": .25, "end": .7, "speaker": 2}]
        cores = [{"start": .1, "end": .3, "speaker": 1}, {"start": .5, "end": .6, "speaker": 2}]
        self.assertEqual(priority_activity(padded, cores), [
            {"start": 0., "end": .3, "speaker": 1, "uncertain": False},
            {"start": .3, "end": .5, "speaker": UNCERTAIN_BOUNDARY, "uncertain": True},
            {"start": .5, "end": .7, "speaker": 2, "uncertain": False}])
        values = array.array("f", [0.] * 8 * 100)
        for frame in range(100):
            values[frame * 8] = .9
            if 20 <= frame < 50:
                values[frame * 8 + 1] = .95
        activity = priority_activity([], posterior_cores(values, .01))
        self.assertEqual({r["speaker"] for r in activity if r["start"] <= .3 < r["end"]}, {1, 2})


if __name__ == "__main__":
    unittest.main()
