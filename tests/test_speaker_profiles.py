from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from workbench.config import load_config
from workbench.speaker_profiles import (
    SpeakerProfileError,
    enroll_speaker,
    list_profiles,
    profile_store_path,
    remove_profile,
)


class SpeakerProfileTests(unittest.TestCase):
    def test_enrollment_requires_explicit_consent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sample = Path(temp_dir) / "sample.wav"
            sample.write_bytes(b"audio")
            with self.assertRaises(SpeakerProfileError):
                enroll_speaker("Alice", [sample], consent=False)

    def test_enrollment_persists_metadata_without_audio_bytes_and_remove_cleans_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample = root / "sample.wav"
            sample.write_bytes(b"PRIVATE AUDIO CONTENT")
            config = load_config(user_path=root / "config.toml")

            def fake_write(path: Path, _embedding: object) -> str:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"PRIVATE EMBEDDING CONTENT")
                return "embedding-digest"

            with patch("workbench.speaker_profiles._package_available", return_value=True):
                with patch("workbench.speaker_profiles._embed_with_resemblyzer", return_value=([0.1, 0.2], "test/VoiceEncoder")):
                    with patch("workbench.speaker_profiles._write_embedding", side_effect=fake_write):
                        profile = enroll_speaker("Alice", [sample], consent=True, config=config)

            store = profile_store_path(config)
            raw = store.read_text(encoding="utf-8")
            payload = json.loads(raw)
            self.assertEqual(profile["person_id"], "person_alice")
            self.assertEqual(profile["sample_count"], 1)
            self.assertTrue(profile["consent"])
            self.assertNotIn("PRIVATE AUDIO CONTENT", raw)
            self.assertNotIn("PRIVATE EMBEDDING CONTENT", raw)
            self.assertTrue(Path(profile["embedding_path"]).exists())
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(list_profiles(config), [profile])

            removed = remove_profile(profile["person_id"], config=config)
            self.assertEqual(removed["status"], "removed")
            self.assertEqual(list_profiles(config), [])
            self.assertFalse(Path(profile["embedding_path"]).exists())

    def test_profile_store_defaults_outside_repository(self) -> None:
        store = profile_store_path()
        self.assertEqual(store.name, "speaker_profiles.json")
        self.assertNotEqual(store.parent, Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    unittest.main()
