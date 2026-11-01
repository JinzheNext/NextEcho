"""Exercise real exports/readers and manual publication without model inference."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app import (
    app,
    load_speaker_transcript,
    _aggregate_speaker_payload,
    _speaker_item_payload,
    _speaker_payload_v3,
)
from workbench.acoustic_dialogue import (
    build_dialogue_payload,
    write_dialogue_outputs,
    load_dialogue_segments,
)
from workbench.contracts import PipelineError
from workbench.io_utils import file_hash, write_json
from workbench.postprocess import load_speaker_turns
from workbench.publication import publish_pipeline, verify_publication
from workbench.speaker_artifacts import dialogue_to_v3, read_speaker_file, write_identity_reading
from tests.test_publication import factory_item


def make_dialogue(folder: Path, *, unanchored=False):
    text = "金额36万元。大家好，我是李继刚。"
    raw = [{"id": "raw-1", "text": text, "start_ms": 0, "end_ms": len(text) * 200}]
    split = text.index("大家好") * 200 / 1000
    tokens = (
        []
        if unanchored
        else [
            {"text": c, "start_ms": i * 200, "end_ms": None} for i, c in enumerate(text)
        ]
    )
    turns = [
        {"speaker": "A", "start": 0, "end": split},
        {"speaker": "B", "start": split, "end": len(text) * 200 / 1000},
    ]
    write_json(
        folder / "transcript.json",
        {
            "transcription": [
                {
                    "id": "raw-1",
                    "text": text,
                    "offsets": {"from": 0, "to": len(text) * 200},
                }
            ]
        },
    )
    data = build_dialogue_payload(
        raw, raw, tokens, turns, {"speaker_candidates": [{"name": "李继刚"}]}
    )
    data["raw_sha256"] = file_hash(folder / "transcript.json")
    paths = write_dialogue_outputs(folder, data)
    write_json(
        folder / "transcript.enhanced.json",
        {
            "enhanced_transcription": [
                {
                    "id": r["id"],
                    "text": r["text"].replace("大家好，", "大家好！"),
                    "speaker": r["speaker"],
                    "offsets": {"from": r["start_ms"], "to": r["end_ms"]},
                }
                for r in data["transcription"]
            ]
        },
    )
    return data, paths


class SpeakerArtifactTests(unittest.TestCase):
    def test_identity_reading_splits_mixed_parent_by_exact_character_spans_and_uuids(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            text = "甲乙丙丁"
            digest = __import__("hashlib").sha256(text.encode()).hexdigest()
            write_json(root / "transcript.enhanced.json", {"enhanced_transcription": [
                {"id": "s", "text": text, "offsets": {"from": 0, "to": 1000}}
            ]})
            payload = {"speaker_turns": [
                {"id": "t0", "speaker": "孟岩", "speaker_id": "spk_00", "speaker_uuid": "voice-a", "segment_ids": ["s"],
                 "alignment_spans": [{"segment_id": "s", "source_text_sha256": digest, "text_start": 0, "text_end": 2,
                                      "assignment_reason": "native_interval", "native_start_ms": 0, "native_end_ms": 400,
                                      "review_start_ms": 0, "review_end_ms": 400}]},
                {"id": "t1", "speaker": "李继刚", "speaker_id": "spk_01", "speaker_uuid": "voice-b", "segment_ids": ["s"],
                 "alignment_spans": [{"segment_id": "s", "source_text_sha256": digest, "text_start": 2, "text_end": 4,
                                      "assignment_reason": "native_interval", "native_start_ms": 400, "native_end_ms": 800,
                                      "review_start_ms": 400, "review_end_ms": 800}]},
            ]}
            result = write_identity_reading(root, payload)
            audit = json.loads(Path(result["identity_json_path"]).read_text())
            rows = audit["transcription"]
            self.assertEqual([(r["text"], r["speaker"], r["speaker_id"]) for r in rows],
                             [("甲乙", "孟岩", "spk_00"), ("丙丁", "李继刚", "spk_01")])
            self.assertEqual([r["speaker_uuid"] for r in rows], ["voice-a", "voice-b"])
            self.assertEqual("".join(r["text"] for r in rows), text)
            self.assertEqual(audit["status"], "ready")

    def test_identity_reading_keeps_same_id_and_name_separate_when_uuids_differ(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            text = "甲乙"
            digest = __import__("hashlib").sha256(text.encode()).hexdigest()
            write_json(root / "transcript.enhanced.json", {"enhanced_transcription": [
                {"id": "s", "text": text, "offsets": {"from": 0, "to": 1000}}
            ]})
            payload = {"speaker_turns": [
                {"id": "a", "speaker": "孟岩", "speaker_id": "spk_00", "speaker_uuid": "voice-a",
                 "segment_ids": ["s"], "alignment_spans": [{"segment_id": "s", "source_text_sha256": digest,
                    "text_start": 0, "text_end": 1, "assignment_reason": "native_interval",
                    "native_start_ms": 100, "native_end_ms": 200}]},
                {"id": "b", "speaker": "孟岩", "speaker_id": "spk_00", "speaker_uuid": "voice-b",
                 "segment_ids": ["s"], "alignment_spans": [{"segment_id": "s", "source_text_sha256": digest,
                    "text_start": 1, "text_end": 2, "assignment_reason": "native_interval",
                    "native_start_ms": 300, "native_end_ms": 400}]},
            ]}
            result = write_identity_reading(root, payload)
            audit = json.loads(Path(result["identity_json_path"]).read_text())
            self.assertEqual([(row["text"], row["speaker_uuid"]) for row in audit["transcription"]],
                             [("甲", "voice-a"), ("乙", "voice-b")])

    def test_zero_width_native_span_cannot_assign_identity_or_name(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            text = "我是孟岩"
            digest = __import__("hashlib").sha256(text.encode()).hexdigest()
            write_json(root / "transcript.enhanced.json", {"enhanced_transcription": [
                {"id": "s", "text": text, "offsets": {"from": 0, "to": 1000}}
            ]})
            payload = {"speaker_turns": [{"id": "a", "speaker": "孟岩", "speaker_id": "spk_00",
                "speaker_uuid": "voice-a", "segment_ids": ["s"], "alignment_spans": [{
                    "segment_id": "s", "source_text_sha256": digest, "text_start": 0, "text_end": len(text),
                    "assignment_reason": "native_interval", "native_start_ms": 200, "native_end_ms": 200,
                    "review_start_ms": 100, "review_end_ms": 500,
                }]}]}
            result = write_identity_reading(root, payload)
            audit = json.loads(Path(result["identity_json_path"]).read_text())
            self.assertEqual(audit["status"], "partial")
            self.assertTrue(all(row["speaker"] == "归属待核对" and not row["speaker_uuid"]
                                for row in audit["transcription"]))
            self.assertTrue(all(all(item["timing_kind"] == "review_only" for item in row["voice_evidence"])
                                for row in audit["transcription"]))

    def test_identity_reading_rejects_spans_from_different_source_text(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            original = "甲乙"
            digest = __import__("hashlib").sha256(original.encode()).hexdigest()
            write_json(root / "transcript.enhanced.json", {"enhanced_transcription": [
                {"id": "s", "text": "甲丙", "offsets": {"from": 0, "to": 1000}}
            ]})
            payload = {"speaker_turns": [{"id": "t", "speaker": "孟岩", "speaker_id": "spk_00",
                "segment_ids": ["s"], "alignment_spans": [{"segment_id": "s", "source_text_sha256": digest,
                    "text_start": 0, "text_end": 2, "assignment_reason": "native_interval"}]}]}
            result = write_identity_reading(root, payload)
            audit = json.loads(Path(result["identity_json_path"]).read_text())
            self.assertEqual(audit["status"], "partial")
            self.assertEqual("".join(r["text"] for r in audit["transcription"]), "甲丙")
            self.assertTrue(all(r["speaker"] == "归属待核对" for r in audit["transcription"]))

    def test_identity_reading_maps_only_unchanged_adopted_text_from_raw_alignment(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            raw_text, enhanced_text = "甲乙丙", "甲乙。丙"
            raw_path = root / "transcript.json"
            write_json(raw_path, {"transcription": [{"id": "s", "text": raw_text,
                "offsets": {"from": 0, "to": 1000}}]})
            write_json(root / "transcript.enhanced.json", {"enhanced_transcription": [{
                "id": "s", "text": enhanced_text, "offsets": {"from": 0, "to": 1000}}]})
            digest = __import__("hashlib").sha256(raw_text.encode()).hexdigest()
            payload = {"transcript_source_path": str(raw_path), "transcript_source_sha256": file_hash(raw_path),
                "speaker_turns": [
                    {"id": "a", "speaker": "孟岩", "speaker_id": "spk_00", "segment_ids": ["s"],
                     "alignment_spans": [{"segment_id": "s", "source_text_sha256": digest, "text_start": 0, "text_end": 2,
                        "assignment_reason": "native_interval", "review_start_ms": 0, "review_end_ms": 400}]},
                    {"id": "b", "speaker": "李继刚", "speaker_id": "spk_01", "segment_ids": ["s"],
                     "alignment_spans": [{"segment_id": "s", "source_text_sha256": digest, "text_start": 2, "text_end": 3,
                        "assignment_reason": "native_interval", "review_start_ms": 400, "review_end_ms": 600}]},
                ]}
            result = write_identity_reading(root, payload)
            audit = json.loads(Path(result["identity_json_path"]).read_text())
            self.assertEqual([(r["text"], r["speaker_id"]) for r in audit["transcription"]],
                             [("甲乙。", "spk_00"), ("丙", "spk_01")])
            self.assertEqual("".join(r["text"] for r in audit["transcription"]), enhanced_text)

    @unittest.skipUnless(shutil.which("node"), "Node.js UI function verification")
    def test_shipped_editor_accepts_real_export_and_does_not_prefill_unknown_as_a_name(
        self,
    ):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_dialogue(root)
            payload = root / "ui-payload.json"
            write_json(
                payload,
                {
                    "run_id": "fixture",
                    "speaker_transcript": load_speaker_transcript(root),
                },
            )
            script = """const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync(process.argv[1],'utf8'),payload=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const start=source.indexOf('function renderSpeakerPanel('),end=source.indexOf('\\nfunction runStatusHeadline(',start);
assert(start>=0&&end>start);
function node(tag,props={}){return {tag,...props,children:[],append(...kids){this.children.push(...kids)}};}
const context={makeNode:node,formatCoverage:()=>'',makeLink:()=>null,artifactLink:()=>'',makeActionButton:()=>node('button')};
vm.runInNewContext(source.slice(start,end),context);
const rendered=context.renderSpeakerPanel(payload),nodes=[];
function walk(n){nodes.push(n);for(const child of n.children||[])walk(child);}walk(rendered);
const inputs=nodes.filter(n=>n.tag==='input'),people=payload.speaker_transcript.speakers;
assert.strictEqual(inputs.length,people.length);
for(let i=0;i<inputs.length;i++){
 assert.strictEqual(inputs[i].dataset.speakerId,people[i].speaker_id);
 assert.strictEqual(inputs[i].attrs.value,people[i].identity_status==='resolved'?people[i].display_name:'');
 assert(nodes.some(n=>n.tag==='span'&&n.text===`${people[i].anonymous_label} (${people[i].speaker_id})`));
}
assert(inputs.some(n=>n.attrs.value===''));assert(inputs.some(n=>n.attrs.value==='李继刚'));
"""
            result = subprocess.run(
                [
                    shutil.which("node"),
                    "-e",
                    script,
                    str(Path(__file__).resolve().parents[1] / "static/app.js"),
                    str(payload),
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_actual_writer_all_files_and_readers_preserve_ids_evidence_and_native_null(
        self,
    ):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            original, paths = make_dialogue(root)
            payload = read_speaker_file(Path(paths["speaker_json_path"]))
            for key in (
                "speaker_json_path",
                "speaker_text_path",
                "speaker_markdown_path",
                "speaker_map_path",
                "dialogue_json_path",
                "dialogue_text_path",
            ):
                self.assertTrue(Path(paths[key]).is_file(), key)
            self.assertIsInstance(payload["speakers"], list)
            self.assertEqual(
                [p["speaker_id"] for p in payload["speakers"]],
                [p["speaker_id"] for p in original["speakers"]["speakers"]],
            )
            self.assertTrue(
                all(
                    span["native_end_ms"] is None
                    for r in payload["speaker_turns"]
                    for span in r["alignment_spans"]
                )
            )
            self.assertEqual(load_speaker_turns(root), payload["speaker_turns"])
            self.assertEqual(
                load_speaker_transcript(root)["speakers"], payload["speakers"]
            )
            self.assertIn("金额36万元", Path(paths["speaker_text_path"]).read_text())
            self.assertTrue(
                any(r["speaker"] == "李继刚" for r in payload["speaker_turns"])
            )
            self.assertFalse(payload["scoring_eligible"])

    def test_old_nested_file_is_read_only_and_unassigned_rows_do_not_create_people(
        self,
    ):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data, _ = make_dialogue(root, unanchored=True)
            data["version"] = "native-ctc-dialogue-v2-evidence"
            path = root / "old.transcript.speakers.json"
            write_json(path, data)
            digest = file_hash(path)
            payload = read_speaker_file(path)
            view = _speaker_payload_v3(payload)
            self.assertEqual(file_hash(path), digest)
            self.assertEqual(len(view["speakers"]), 2)
            self.assertFalse(view["speaker_turns"][0]["speaker_id"])
            self.assertNotIn(
                "归属待核对", [p["display_name"] for p in view["speakers"]]
            )

    def test_tampered_export_or_invalid_protocol_cannot_be_silently_repaired(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data, paths = make_dialogue(root)
            path = Path(paths["speaker_json_path"])
            payload = json.loads(path.read_text())
            payload["speakers"][0]["display_name"] = "伪造姓名"
            write_json(path, payload)
            with self.assertRaises(PipelineError) as caught:
                read_speaker_file(path)
            self.assertEqual(caught.exception.code, "CACHE_MISMATCH")
            data["transcription"][0]["speaker_id"] = "not-in-recording"
            with self.assertRaises(PipelineError):
                dialogue_to_v3(data)

    def prepare_run(self, root, names=("one",)):
        results = []
        views = []
        for name in names:
            item = factory_item(root, name)
            folder = Path(item["output_dir"])
            _, paths = make_dialogue(folder)
            item.update(paths)
            item["speaker_status"] = "ready"
            results.append(item)
            views.append(_speaker_item_payload(folder, name))
        _aggregate_speaker_payload(root, views)
        receipt = publish_pipeline(
            root,
            job_id="base",
            results=results,
            stages={"asr": "completed", "speaker": "completed", "polish": "completed"},
            elapsed_seconds=77,
        )
        return receipt

    def test_manual_api_creates_verified_revision_and_keeps_entire_base_unchanged(self):
        with tempfile.TemporaryDirectory() as d, patch("app.RUNS_ROOT", Path(d)):
            root = Path(d) / "base"
            old_receipt = self.prepare_run(root)
            source = load_speaker_transcript(root)
            identity = source["speakers"][0]["speaker_id"]
            before = {
                p.relative_to(root): file_hash(p)
                for p in root.rglob("*")
                if p.is_file()
            }
            client = app.test_client()
            response = client.post(
                "/api/runs/base/speaker-map", json={"mapping": {identity: "孟岩"}}
            )
            self.assertEqual(response.status_code, 200, response.get_json())
            value = response.get_json()
            revised = Path(d) / value["run_id"]
            self.assertNotEqual(revised, root)
            self.assertEqual(
                {
                    p.relative_to(root): file_hash(p)
                    for p in root.rglob("*")
                    if p.is_file()
                },
                before,
            )
            self.assertEqual(verify_publication(root), old_receipt)
            receipt = verify_publication(revised)
            self.assertIn("孟岩：", Path(receipt["primary_file"]).read_text())
            self.assertIn("大家好！", Path(receipt["primary_file"]).read_text())
            manifest = json.loads((revised / "manifest.json").read_text())
            self.assertEqual(
                manifest["speaker_identity"]["speakers"][0]["display_name"], "孟岩"
            )
            self.assertEqual(
                manifest["speaker_transcript"]["speakers"][0]["display_name"], "孟岩"
            )
            self.assertEqual(
                manifest["results"][0]["speaker_identity"]["speakers"][0][
                    "display_name"
                ],
                "孟岩",
            )
            self.assertTrue(
                Path(manifest["speaker_identity"]["mapping_path"]).is_file()
            )
            new_item = revised / "items" / "one"
            self.assertEqual(
                file_hash(new_item / "transcript.enhanced.json"),
                before[Path("items/one/transcript.enhanced.json")],
            )
            rows, data = load_dialogue_segments(
                new_item, file_hash(new_item / "transcript.json")
            )
            self.assertEqual(rows[0]["speaker"], "孟岩")
            self.assertIn("identity_mapping_sha256", data)
            self.assertEqual(
                read_speaker_file(new_item / "transcript.speakers.json")[
                    "speaker_turns"
                ],
                rows_with_seconds(rows),
            )
            self.assertEqual(receipt["plan"]["models_called"], 0)
            self.assertLess(
                json.loads((revised / "manifest.json").read_text())["elapsed_seconds"],
                77,
            )
            bad = client.post(
                "/api/runs/base/speaker-map", json={"mapping": {"missing": "错误"}}
            )
            self.assertEqual(bad.status_code, 400)
            # Editing the revision's mapping is detected by the publication
            # ledger; original delivery remains independently valid.
            (new_item / "speaker.identity-map.json").write_text("{}")
            with self.assertRaises(PipelineError):
                verify_publication(revised)
            self.assertEqual(verify_publication(root), old_receipt)

    def test_duplicate_local_ids_in_two_recordings_are_scoped_and_clearing_is_explicit(
        self,
    ):
        with tempfile.TemporaryDirectory() as d, patch("app.RUNS_ROOT", Path(d)):
            root = Path(d) / "base"
            self.prepare_run(root, ("one", "two"))
            source = load_speaker_transcript(root)
            ids = [p["speaker_id"] for p in source["speakers"]]
            self.assertEqual(len(ids), len(set(ids)))
            root_map = json.loads((root / "speaker_map.json").read_text())
            self.assertTrue(root_map)
            self.assertTrue(all(key in ids and key for key in root_map))
            client = app.test_client()
            response = client.post(
                "/api/runs/base/speaker-map", json={"mapping": {ids[0]: "孟岩"}}
            )
            self.assertEqual(response.status_code, 200, response.get_json())
            value = response.get_json()
            revised = Path(d) / value["run_id"]
            one = read_speaker_file(revised / "items/one/transcript.speakers.json")
            two = read_speaker_file(revised / "items/two/transcript.speakers.json")
            self.assertEqual(one["speakers"][0]["display_name"], "孟岩")
            self.assertEqual(two["speakers"][0]["identity_status"], "unknown")
            cleared = client.post(
                "/api/runs/" + value["run_id"] + "/speaker-map",
                json={"mapping": {ids[0]: ""}},
            )
            self.assertEqual(cleared.status_code, 200, cleared.get_json())
            newest = Path(d) / cleared.get_json()["run_id"]
            one = read_speaker_file(newest / "items/one/transcript.speakers.json")
            self.assertEqual(one["speaker_turns"][0]["speaker"], "说话人 1")
            self.assertEqual(one["speakers"][0]["identity_status"], "unknown")
            manifest = json.loads((newest / "manifest.json").read_text())
            self.assertEqual(
                manifest["results"][0]["speaker_identity"]["speakers"][0][
                    "identity_status"
                ],
                "unknown",
            )

    def test_changed_aggregate_cannot_authorize_a_fictional_person_and_missing_item_does_not_borrow(
        self,
    ):
        with tempfile.TemporaryDirectory() as d, patch("app.RUNS_ROOT", Path(d)):
            root = Path(d) / "base"
            self.prepare_run(root, ("one", "two"))
            path = root / "transcript.speakers.json"
            payload = json.loads(path.read_text())
            payload["speakers"][0]["speaker_id"] = "invented"
            write_json(path, payload)
            response = app.test_client().post(
                "/api/runs/base/speaker-map", json={"mapping": {"invented": "虚构人物"}}
            )
            self.assertEqual(response.status_code, 409, response.get_json())
            self.assertEqual(len(list(Path(d).iterdir())), 1)
            (root / "items/one/transcript.speakers.json").unlink()
            with self.assertRaises(PipelineError):
                load_speaker_turns(root / "items/one")


def rows_with_seconds(rows):
    result = copy.deepcopy(rows)
    for row in result:
        row.update(
            start=row["start_ms"] / 1000,
            end=row["end_ms"] / 1000,
            annotation_kind="text-review-envelope",
            timing_precision="review-envelope",
        )
    return result


if __name__ == "__main__":
    unittest.main()
