"""Recording/stage IDs across copies/reorders and real manual publication."""

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import wave

from app import app, load_speaker_transcript
from workbench.acoustic_dialogue import write_dialogue_outputs
from workbench.contracts import PipelineError
from workbench.io_utils import file_hash, write_json
from workbench.publication import publish_pipeline, verify_publication
from workbench.speaker_artifacts import read_speaker_file
from workbench.speaker_scope import bind_recording_scope, LEGACY_SCOPE, SCOPE_VERSION
from workbench.speaker_service import _aggregate_speaker_payload, _speaker_item_payload
from tests.test_publication import factory_item
from tests.test_speaker_artifacts import make_dialogue


def strong_fixture(folder, source_id):
    folder.mkdir(parents=True, exist_ok=True)
    audio = folder / "audio.wav"
    with wave.open(str(audio), "wb") as f:
        f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        f.writeframes(b"\x01\x00" * 16000 * 8)
    data, _ = make_dialogue(folder)
    data["audio_sha256"] = file_hash(audio)
    data["provenance"] = {
        "speaker_models": {"/local/models/segmentation.onnx": "1" * 64},
        "runtime": {"packages": {"synthetic-runtime": "1"}},
    }
    write_json(folder / "metadata.json", {"source_id": source_id})
    paths = write_dialogue_outputs(folder, data)
    return data, paths


class SpeakerScopeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="声音范围 ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_file_reader_binds_actual_audio_and_retains_local_ids_without_writing(self):
        folder = self.root / "one"
        data, paths = strong_fixture(folder, "episode:one")
        before = {p.name: file_hash(p) for p in folder.iterdir() if p.is_file()}
        view = read_speaker_file(Path(paths["speaker_json_path"]))
        self.assertEqual(
            view["recording_identity"]["status"], "verified-recording-stage"
        )
        self.assertTrue(view["recording_identity"]["model_identity_known"])
        self.assertEqual(
            [p["speaker_id"] for p in view["speakers"]],
            [p["speaker_id"] for p in data["speakers"]["speakers"]],
        )
        self.assertTrue(all(p.get("speaker_uuid") for p in view["speakers"]))
        by_id = {p["speaker_id"]: p["speaker_uuid"] for p in view["speakers"]}
        self.assertTrue(
            all(
                not r["speaker_id"] or r["speaker_uuid"] == by_id[r["speaker_id"]]
                for r in view["speaker_turns"]
            )
        )
        self.assertEqual(
            before, {p.name: file_hash(p) for p in folder.iterdir() if p.is_file()}
        )

    def test_copy_paths_names_text_and_local_renumbering_do_not_define_voice_uuid(self):
        folder = self.root / "one"
        _, paths = strong_fixture(folder, "episode:one")
        original = read_speaker_file(Path(paths["speaker_json_path"]))
        duplicate = self.root / "copied"
        shutil.copytree(folder, duplicate)
        copied = read_speaker_file(duplicate / "transcript.speakers.json")
        self.assertEqual(original["recording_identity"], copied["recording_identity"])
        edited = copy.deepcopy(original)
        mapping = {}
        for index, person in enumerate(edited["speakers"]):
            mapping[person["speaker_id"]] = "changed_" + str(index)
            person["speaker_id"] = mapping[person["speaker_id"]]
            person["display_name"] = "人工新姓名"
        for row in edited["speaker_turns"]:
            if row["speaker_id"]:
                row["speaker_id"] = mapping[row["speaker_id"]]
            row["text"] += "文字校订"
        edited["raw_sha256"] = "9" * 64
        edited["provenance"]["speaker_models"] = {
            "/new/install/segmentation.onnx": "1" * 64
        }
        rebound = bind_recording_scope(
            edited,
            source_id="episode:one",
            audio_sha256=original["audio_sha256"],
            audio_frames=128000,
            sample_rate=16000,
        )
        self.assertEqual(
            [p["speaker_uuid"] for p in original["speakers"]],
            [p["speaker_uuid"] for p in rebound["speakers"]],
        )

    def test_source_audio_activity_and_known_model_changes_are_separate_revisions(self):
        folder = self.root / "one"
        _, paths = strong_fixture(folder, "episode:one")
        original = read_speaker_file(Path(paths["speaker_json_path"]))
        voices = [p["speaker_uuid"] for p in original["speakers"]]
        for source_id, digest, change in [
            ("episode:two", original["audio_sha256"], None),
            ("episode:one", "2" * 64, None),
            ("episode:one", original["audio_sha256"], "activity"),
            ("episode:one", original["audio_sha256"], "model"),
        ]:
            data = copy.deepcopy(original)
            if change == "activity":
                data["acoustic_activity"]["turns"][0]["end"] -= 0.01
            if change == "model":
                data["provenance"]["speaker_models"][
                    "/local/models/segmentation.onnx"
                ] = "3" * 64
            result = bind_recording_scope(
                data,
                source_id=source_id,
                audio_sha256=digest,
                audio_frames=128000,
                sample_rate=16000,
            )
            self.assertNotEqual(voices, [p["speaker_uuid"] for p in result["speakers"]])

    def test_reordering_and_relabeling_recordings_preserves_global_keys(self):
        entries = []
        for label in ("one", "two"):
            folder = self.root / label
            strong_fixture(folder, "episode:" + label)
            entries.append(_speaker_item_payload(folder, label))
        first = _aggregate_speaker_payload(self.root, entries, write_files=False)
        changed = copy.deepcopy(entries[::-1])
        changed[0]["source_label"] = "new-label"
        second = _aggregate_speaker_payload(self.root, changed, write_files=False)
        self.assertEqual(first["identity_scope_version"], SCOPE_VERSION)
        self.assertEqual(
            {p["speaker_id"] for p in first["speakers"]},
            {p["speaker_id"] for p in second["speakers"]},
        )
        self.assertEqual(
            len(first["speakers"]), len({p["speaker_id"] for p in first["speakers"]})
        )

    def test_repeated_same_recording_is_rejected_instead_of_ambiguous_names(self):
        folder = self.root / "one"
        strong_fixture(folder, "episode:one")
        view = _speaker_item_payload(folder, "one")
        with self.assertRaises(PipelineError):
            _aggregate_speaker_payload(
                self.root, [view, copy.deepcopy(view)], write_files=False
            )

    def test_unverified_old_backend_local_ids_are_isolated_in_new_multi_source_runs(self):
        legacy = {"schema_version": 3, "status": "partial",
            "speakers": [{"speaker_id": "SPEAKER_00", "raw_labels": ["SPEAKER_00"],
                "display_name": "unknown", "identity_status": "unknown", "identity_evidence": [], "confidence": 0}],
            "speaker_turns": [{"speaker_id": "SPEAKER_00", "speaker": "说话人 1", "text": "原话",
                "start": 0, "end": 1}], "recording_identity": {"status": "unverified-legacy"}}
        entries = [{"source_label": label, "payload": copy.deepcopy(legacy)} for label in ("one", "two")]
        aggregate = _aggregate_speaker_payload(self.root, entries, write_files=False)
        self.assertEqual(len(aggregate["speakers"]), 2)
        self.assertEqual(len({p["speaker_id"] for p in aggregate["speakers"]}), 2)
        self.assertFalse(any(p.get("speaker_uuid") for p in aggregate["speakers"]))

    def test_legacy_projection_cannot_mutate_shared_current_identity_views(self):
        folder = self.root / "one"
        strong_fixture(folder, "episode:one")
        entries = [_speaker_item_payload(folder, "one")]
        original = copy.deepcopy(entries)
        _aggregate_speaker_payload(self.root, entries, write_files=False, identity_scope_version=LEGACY_SCOPE)
        self.assertEqual(entries, original)
        current = _aggregate_speaker_payload(self.root, entries, write_files=False)
        self.assertTrue(current["speakers"][0]["speaker_uuid"])

    def test_legacy_missing_audio_hash_is_explicit_and_old_aggregate_remains_readable(
        self,
    ):
        folder = self.root / "old" / "items" / "one"
        folder.mkdir(parents=True)
        _, paths = make_dialogue(folder)
        view = read_speaker_file(Path(paths["speaker_json_path"]))
        self.assertEqual(view["recording_identity"]["status"], "unverified-legacy")
        self.assertFalse(any("speaker_uuid" in p for p in view["speakers"]))
        run = folder.parents[1]
        _aggregate_speaker_payload(
            run,
            [_speaker_item_payload(folder, "one")],
            identity_scope_version=LEGACY_SCOPE,
        )
        before = file_hash(run / "transcript.speakers.json")
        self.assertTrue(load_speaker_transcript(run)["speakers"])
        self.assertEqual(before, file_hash(run / "transcript.speakers.json"))

    def test_changed_source_or_audio_cannot_validate_persisted_uuid(self):
        folder = self.root / "one"
        _, paths = strong_fixture(folder, "episode:one")
        source = Path(paths["speaker_json_path"])
        scoped = read_speaker_file(source)
        write_json(source, scoped)
        self.assertEqual(
            read_speaker_file(source)["recording_identity"],
            scoped["recording_identity"],
        )
        write_json(folder / "metadata.json", {"source_id": "episode:other"})
        with self.assertRaises(PipelineError):
            read_speaker_file(source)

        write_json(folder / "metadata.json", {"source_id": "episode:one"})
        (folder / "audio.wav").write_bytes(b"corrupt wav")
        with self.assertRaises(PipelineError):
            read_speaker_file(source)

    def test_legacy_reference_audio_hash_is_not_reinterpreted_without_acoustic_binding(self):
        folder = self.root / "legacy"
        data, paths = strong_fixture(folder, "episode:one")
        data.pop("acoustic_activity")
        data["audio_sha256"] = "0" * 64
        # Old exports can refer to a processed voice reference rather than
        # this normalized WAV; readable, but never a new verified UUID.
        write_dialogue_outputs(folder, data)
        before = file_hash(Path(paths["speaker_json_path"]))
        view = read_speaker_file(Path(paths["speaker_json_path"]))
        self.assertEqual(view["recording_identity"]["status"], "unverified-legacy")
        self.assertFalse(any(p.get("speaker_uuid") for p in view["speakers"]))
        self.assertEqual(before, file_hash(Path(paths["speaker_json_path"])))

    def test_new_aggregate_public_mapping_tamper_is_detected(self):
        run = self.root / "run"
        folder = run / "items" / "one"
        strong_fixture(folder, "episode:one")
        _aggregate_speaker_payload(run, [_speaker_item_payload(folder, "one")])
        path = run / "transcript.speakers.json"
        data = json.loads(path.read_text())
        key = next(iter(data["items"][0]["public_speaker_ids"]))
        data["items"][0]["public_speaker_ids"][key] = "wrong-source/wrong-voice"
        write_json(path, data)
        with self.assertRaises(PipelineError):
            load_speaker_transcript(run)

    def test_manual_api_keeps_scope_and_updates_only_selected_recording(self):
        run = self.root / "base"
        run.mkdir()
        items, views = [], []
        for label in ("one", "two"):
            item = factory_item(run, label)
            folder = Path(item["output_dir"])
            strong_fixture(folder, "episode:" + label)
            item["source_id"] = "episode:" + label
            write_json(folder / "metadata.json", item)
            view = _speaker_item_payload(folder, label)
            item.update(view["artifact_paths"])
            items.append(item)
            views.append(view)
        _aggregate_speaker_payload(run, views)
        items[0]["aggregate_speaker_json_path"] = str(run / "transcript.speakers.json")
        write_json(run / "manifest.json", {"results": items})
        publish_pipeline(run, job_id="scope-base", results=items, elapsed_seconds=1)
        old = {
            str(p.relative_to(run)): file_hash(p) for p in run.rglob("*") if p.is_file()
        }
        original = load_speaker_transcript(run)
        selected = original["speakers"][0]
        with patch("app.RUNS_ROOT", self.root):
            response = app.test_client().post(
                "/api/runs/base/speaker-map",
                json={"mapping": {selected["speaker_id"]: "孟岩"}},
            )
        self.assertEqual(response.status_code, 200, response.get_json())
        revision = self.root / response.get_json()["run_id"]
        revised = load_speaker_transcript(revision)
        self.assertEqual(
            {p["speaker_uuid"] for p in original["speakers"]},
            {p["speaker_uuid"] for p in revised["speakers"]},
        )
        self.assertEqual(revised["speakers"][0]["display_name"], "孟岩")
        self.assertNotEqual(revised["speakers"][2]["display_name"], "孟岩")
        self.assertIsNotNone(verify_publication(revision))
        self.assertTrue(all(file_hash(run / name) == sha for name, sha in old.items()))


if __name__ == "__main__":
    unittest.main()
