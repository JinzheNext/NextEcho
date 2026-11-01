import json
import tempfile
import unittest
import wave
from pathlib import Path

from workbench.identity_review import write_identity_review
from workbench.io_utils import file_hash, write_json


class IdentityReviewTests(unittest.TestCase):
    def test_private_audio_link_and_hostile_text_remain_local_inert_data(self):
        hostile = '</script><img src="https://example.org/x" onerror="alert(1)">'
        with tempfile.TemporaryDirectory(prefix="身份审核 ") as d:
            root = Path(d)
            audio = root / "原声.wav"
            with wave.open(str(audio), "wb") as f:
                f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                f.writeframes(b"\0\0" * 16000)
            dialogue = root / "dialogue.json"
            write_json(dialogue, {"transcription": [{"id": "s1", "text": hostile, "start_ms": 0, "end_ms": 1000}],
                "speakers": {"speakers": []}, "speaker_labels": {}, "alignment_audit": {}, "raw_sha256": "raw",
                "audio_sha256": file_hash(audio)})
            page = root / "review" / "identity.html"
            write_identity_review(page, dialogue, audio, audio_origin_ms=0)
            body = page.read_text()
            embedded = body.split('<script id="identity-data" type="application/json">')[1].split('</script>')[0]
            data = json.loads(embedded)
            self.assertEqual(data["rows"][0]["text"], hostile)
            self.assertNotIn("<img", embedded)
            self.assertTrue(data["audio"].startswith("../"))
            self.assertNotIn("http", data["audio"])
            self.assertEqual(data["source"]["quality_status"], "human_review_pending")

    def test_duplicate_ids_are_rejected_before_saving_review(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "dialogue.json"
            write_json(source, {"transcription": [{"id": "s1"}, {"id": "s1"}]})
            with self.assertRaises(ValueError):
                write_identity_review(root / "review.html", source, root / "audio.wav")

    def test_identity_page_separates_unknown_name_overlap_and_missing_anchor(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            audio = root / "source.wav"
            with wave.open(str(audio), "wb") as f:
                f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                f.writeframes(b"\0\0" * 32000)
            source = root / "dialogue.json"
            write_json(source, {
                "transcription": [
                    {"id": "unassigned", "text": "未知", "speaker": "归属待核对", "speaker_id": "",
                     "identity_status": "unassigned", "segment_ids": ["s1"], "start_ms": 100, "end_ms": 500,
                     "native_anchor_chars": 0, "alignment_reason_counts": [{"reason": "no_acoustic_activity", "count": 2}]},
                    {"id": "anonymous", "text": "有声音", "speaker": "说话人 1", "speaker_id": "spk_00",
                     "speaker_uuid": "speaker-uuid", "identity_status": "unknown", "segment_ids": ["s2"],
                     "start_ms": 600, "end_ms": 1000, "native_anchor_chars": 2,
                     "alignment_reason_counts": [{"reason": "native_onset", "count": 2}]},
                    {"id": "overlap", "text": "交叠", "speaker": "交叠发言", "speaker_id": "",
                     "identity_status": "unassigned", "segment_ids": ["s3"], "start_ms": 1100, "end_ms": 1500,
                     "native_anchor_chars": 0, "alignment_reason_counts": [{"reason": "acoustic_overlap", "count": 2}]},
                ],
                "speakers": {"speakers": [{"speaker_id": "spk_00", "speaker_uuid": "speaker-uuid",
                    "anonymous_label": "说话人 1", "raw_labels": [], "identity_status": "unknown"}]},
                "speaker_labels": {}, "alignment_audit": {}, "raw_sha256": "raw",
                "audio_sha256": file_hash(audio), "audio_origin_ms": 0,
                "text_layer": "historical-enhanced-replay-not-new-polish",
                "speaker_payload_sha256": "a" * 64, "recording_id": "recording-uuid",
            })
            page = root / "review" / "identity.html"
            write_identity_review(page, source, audio, audio_origin_ms=0)
            body = page.read_text()
            embedded = body.split('<script id="identity-data" type="application/json">')[1].split('</script>')[0]
            data = json.loads(embedded)
            self.assertIn('value="unknown-speaker"', body)
            self.assertIn('value="unknown-name"', body)
            self.assertIn('value="overlap"', body)
            self.assertIn('value="no-anchor"', body)
            self.assertIn('value="missing-duration"', body)
            self.assertIn("historical-enhanced-replay-not-new-polish", body)
            self.assertEqual(data["review_items"][0]["source"]["speaker_payload_sha256"], "a" * 64)
            self.assertEqual(data["review_items"][0]["source"]["recording_id"], "recording-uuid")
