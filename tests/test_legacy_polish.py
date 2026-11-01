"""Owned toy processes through the actual compatibility adapter; no real model."""

from __future__ import annotations

import json
import os
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from workbench.config import NextEchoConfig
from workbench.contracts import PipelineError
from workbench.enhancement import (
    maybe_ai_polish,
    default_glossary_rules,
    enhance_transcript_artifacts,
)
from workbench.io_utils import file_hash, write_json
from workbench.resources import GIB


def sample():
    return {"available_bytes": 100 * GIB, "swap_used_bytes": 0, "pressure": "normal"}


def config(**changes):
    return NextEchoConfig(
        {
            "hardware": {"reserve_memory_gb": 0},
            "postprocess": {
                "legacy_command": {
                    "estimated_bytes": 32 * 1024**2,
                    "max_rss_bytes": 512 * 1024**2,
                    "timeout_seconds": 5,
                    "max_output_bytes": 65536,
                    **changes,
                }
            },
        }
    )


class LegacyPolishTests(unittest.TestCase):
    def program(self, root, output="", body=None):
        path = root / "toy model 中文.py"
        path.write_text(
            body
            or "import sys,json\njson.load(sys.stdin)\nsys.stdout.write("
            + repr(output)
            + ")\n"
        )
        return shlex.join([sys.executable, str(path)])

    def context(self, root, command):
        return patch.dict(
            os.environ,
            {
                "NEXTECHO_RUNTIME_DIR": str(root / "runtime"),
                "NEXTECHO_JOB_ID": "",
                "TRANSCRIBE_AI_POLISH_COMMAND": command,
            },
        )

    def source(self, root, rows):
        raw = {"transcription": rows}
        write_json(root / "transcript.json", raw)
        (root / "transcript.txt").write_text("\n\n".join(row["text"] for row in rows))
        return raw, {
            name: file_hash(root / name)
            for name in ("transcript.json", "transcript.txt")
        }

    @unittest.skipUnless(os.name == "posix", "owned POSIX command")
    def test_direct_helper_rejects_amount_polarity_and_records_independent_candidate(
        self,
    ):
        with (
            tempfile.TemporaryDirectory() as d,
            patch("workbench.media_commands.sample_resources", side_effect=sample),
        ):
            root = Path(d)
            text = "金额36万元，不能超过限额。"
            candidate = "金额37万元，可以超过限额。"
            with self.context(root, self.program(root, candidate)):
                output, audit = maybe_ai_polish(
                    text,
                    rules=default_glossary_rules(""),
                    enabled=True,
                    config=config(),
                )
            self.assertEqual(output, text)
            self.assertFalse(audit["applied"])
            self.assertEqual(audit["status"], "partial")
            self.assertFalse(audit["protected_fields_pass"])
            self.assertEqual(Path(audit["candidate"]["path"]).read_text(), candidate)
            self.assertEqual(
                file_hash(Path(audit["candidate"]["path"])),
                audit["candidate"]["sha256"],
            )
            self.assertEqual(Path(audit["adopted"]["path"]).read_text(), text)
            self.assertTrue(audit["candidate_patch"]["edits"])
            self.assertFalse(audit["adopted_patch"]["edits"])
            self.assertGreaterEqual(audit["resource_budget"]["reserve_bytes"], 3 * GIB)

    @unittest.skipUnless(os.name == "posix", "owned POSIX command")
    def test_positive_edit_and_equivalent_quantity_are_adopted_as_pending_review(self):
        with (
            tempfile.TemporaryDirectory() as d,
            patch("workbench.media_commands.sample_resources", side_effect=sample),
        ):
            root = Path(d)
            for before, after in [
                ("欢迎来到这里。", "欢迎来到这里！"),
                ("这笔钱是一万元。", "这笔钱是1万元。"),
            ]:
                with (
                    self.subTest(before=before),
                    self.context(root, self.program(root, after)),
                ):
                    text, audit = maybe_ai_polish(
                        before,
                        rules=default_glossary_rules(""),
                        enabled=True,
                        config=config(),
                    )
                self.assertEqual(text, after)
                self.assertTrue(audit["applied"])
                self.assertTrue(audit["protected_fields_pass"])
                self.assertEqual(audit["status"], "partial")
                self.assertTrue(audit["review_required"])
                self.assertEqual(audit["semantic_review"], "pending")

    @unittest.skipUnless(os.name == "posix", "owned POSIX command")
    def test_segment_identity_speaker_and_times_remain_exact_for_adopted_edits(self):
        with (
            tempfile.TemporaryDirectory() as d,
            patch("workbench.media_commands.sample_resources", side_effect=sample),
        ):
            root = Path(d)
            rows = [
                {
                    "id": "a",
                    "text": "欢迎来到这里。",
                    "speaker": "说话人 1",
                    "offsets": {"from": 0, "to": 2000},
                },
                {
                    "id": "b",
                    "text": "金额36万元。",
                    "speaker": "说话人 2",
                    "offsets": {"from": 2000, "to": 4000},
                },
            ]
            raw, before = self.source(root, rows)
            with self.context(
                root, self.program(root, "欢迎来到这里！\n\n金额36万元。")
            ):
                result = enhance_transcript_artifacts(
                    root, ai_polish=True, config=config()
                )
            saved = json.loads((root / "transcript.enhanced.json").read_text())
            self.assertEqual(saved["transcription"], raw["transcription"])
            actual = saved["enhanced_transcription"]
            self.assertEqual(actual[0], {**rows[0], "text": "欢迎来到这里！"})
            self.assertEqual(actual[1], rows[1])
            self.assertEqual({name: file_hash(root / name) for name in before}, before)
            audit = json.loads(Path(result["audit_path"]).read_text())
            self.assertEqual(len(audit["adopted_patches"]), 1)
            self.assertTrue(audit["row_identity_and_times_preserved"])
            self.assertTrue(audit["protected_fields_pass"])
            self.assertEqual(result["status"], "partial")

    @unittest.skipUnless(os.name == "posix", "owned POSIX command")
    def test_rejection_restores_original_before_normalization_and_cross_row_reflow_is_not_guessed(
        self,
    ):
        with (
            tempfile.TemporaryDirectory() as d,
            patch("workbench.media_commands.sample_resources", side_effect=sample),
        ):
            root = Path(d)
            rows = [
                {
                    "id": "a",
                    "text": "这是波客，金额36万元。",
                    "offsets": {"from": 0, "to": 2000},
                },
                {
                    "id": "b",
                    "text": "不能超过限额。",
                    "offsets": {"from": 2000, "to": 4000},
                },
            ]
            _, before = self.source(root, rows)
            for output in [
                "这是播客，金额37万元。\n\n可以超过限额。",
                "这是播客，金额36万元。不能超过限额。",
            ]:
                with (
                    self.subTest(output=output),
                    self.context(root, self.program(root, output)),
                ):
                    result = enhance_transcript_artifacts(
                        root, ai_polish=True, config=config()
                    )
                saved = json.loads((root / "transcript.enhanced.json").read_text())
                self.assertEqual(saved["enhanced_transcription"], rows)
                self.assertEqual(result["status"], "partial")
                self.assertFalse(result["ai"]["applied"])
                self.assertEqual(
                    result["ai"]["fallback_input_layer"], "original-selected-input"
                )
                self.assertEqual(
                    {name: file_hash(root / name) for name in before}, before
                )

    def test_risky_glossary_is_rejected_and_deterministic_candidate_has_patch_and_hash(
        self,
    ):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = [
                {
                    "id": "a",
                    "text": "金额36万元，不能超限。",
                    "offsets": {"from": 0, "to": 1000},
                }
            ]
            self.source(root, rows)
            result = enhance_transcript_artifacts(
                root, user_glossary_text="36 => 37\n不能 => 可以"
            )
            saved = json.loads((root / "transcript.enhanced.json").read_text())
            self.assertEqual(saved["enhanced_transcription"], rows)
            self.assertEqual(result["status"], "partial")
            audit = json.loads(Path(result["audit_path"]).read_text())
            self.assertEqual(
                file_hash(Path(audit["normalization_path"])),
                audit["normalization_sha256"],
            )
            edits = json.loads(Path(audit["normalization_path"]).read_text())
            self.assertTrue(edits[0]["rejected"])
            self.assertTrue(edits[0]["candidate_patch"]["edits"])
            self.assertFalse(edits[0]["adopted_patch"]["edits"])

    def test_missing_budget_blocks_before_process_and_missing_command_is_partial(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with (
                self.context(root, "synthetic-noexec"),
                patch(
                    "workbench.media_commands.run_media_command",
                    side_effect=AssertionError("must not run"),
                ),
            ):
                with self.assertRaises(PipelineError) as caught:
                    maybe_ai_polish(
                        "原文",
                        rules=default_glossary_rules(""),
                        enabled=True,
                        config=NextEchoConfig({}),
                    )
            self.assertEqual(caught.exception.code, "RESOURCE_BLOCKED")
            with self.context(root, ""):
                text, audit = maybe_ai_polish(
                    "原文", rules=default_glossary_rules(""), enabled=True
                )
            self.assertEqual(text, "原文")
            self.assertEqual(audit["status"], "partial")

    @unittest.skipUnless(os.name == "posix", "owned POSIX command")
    def test_timeout_and_output_flood_propagate_without_publishing_enhanced_files(self):
        with (
            tempfile.TemporaryDirectory() as d,
            patch("workbench.media_commands.sample_resources", side_effect=sample),
        ):
            root = Path(d)
            self.source(root, [{"id": "a", "text": "原文"}])
            cases = [
                (
                    "import time;time.sleep(10)",
                    {"timeout_seconds": 0.3},
                    "STAGE_TIMEOUT",
                ),
                (
                    'import sys;sys.stdout.write("x"*1000000)',
                    {"max_output_bytes": 1024},
                    "OUTPUT_LIMIT",
                ),
            ]
            for body, limits, expected in cases:
                with (
                    self.subTest(expected=expected),
                    self.context(root, self.program(root, body=body)),
                ):
                    with self.assertRaises(PipelineError) as caught:
                        enhance_transcript_artifacts(
                            root, ai_polish=True, config=config(**limits)
                        )
                self.assertEqual(caught.exception.code, expected)
                self.assertFalse((root / "transcript.enhanced.txt").exists())
                self.assertFalse((root / "transcript.enhanced.json").exists())

    @unittest.skipUnless(os.name == "posix", "owned POSIX command")
    def test_cancellation_propagates_before_command_and_stderr_stays_private(self):
        with (
            tempfile.TemporaryDirectory() as d,
            patch("workbench.media_commands.sample_resources", side_effect=sample),
        ):
            root = Path(d)
            self.source(root, [{"id": "a", "text": "原文"}])
            with (
                self.context(root, self.program(root, "原文")),
                patch("workbench.media_commands.current_cancelled", return_value=True),
            ):
                with self.assertRaises(PipelineError) as caught:
                    enhance_transcript_artifacts(root, ai_polish=True, config=config())
            self.assertEqual(caught.exception.code, "CANCELLED")
            self.assertFalse((root / "transcript.enhanced.json").exists())
            body = (
                'import sys;sys.stderr.write("token=private-test-secret");sys.exit(2)'
            )
            with self.context(root, self.program(root, body=body)):
                text, audit = maybe_ai_polish(
                    "原文",
                    rules=default_glossary_rules(""),
                    enabled=True,
                    config=config(),
                )
            self.assertEqual(text, "原文")
            self.assertEqual(audit["status"], "partial")
            self.assertNotIn("private-test-secret", json.dumps(audit))
            self.assertIn(
                "private-test-secret", Path(audit["stderr"]["path"]).read_text()
            )

    @unittest.skipUnless(os.name == "posix", "owned POSIX command")
    def test_published_candidate_and_normalization_files_are_verified_after_delivery(
        self,
    ):
        from tests.test_publication import factory_item
        from workbench.publication import publish_pipeline, verify_publication

        with (
            tempfile.TemporaryDirectory() as d,
            patch("workbench.media_commands.sample_resources", side_effect=sample),
        ):
            root = Path(d) / "run"
            item = factory_item(root)
            folder = Path(item["output_dir"])
            self.source(folder, [{"id": "a", "text": "欢迎来到这里。"}])
            with self.context(Path(d), self.program(Path(d), "欢迎来到这里！")):
                result = enhance_transcript_artifacts(
                    folder, ai_polish=True, config=config()
                )
            item.update(
                enhancement=result,
                enhanced_text_path=result["enhanced_text_path"],
                enhanced_json_path=result["enhanced_json_path"],
            )
            receipt = publish_pipeline(root, job_id="legacy-evidence", results=[item])
            candidate = Path(result["ai"]["candidate"]["artifact_path"])
            audit = json.loads(Path(result["audit_path"]).read_text())
            self.assertEqual(receipt["status"], "partial")
            self.assertIn(str(candidate), receipt["artifact_hashes"])
            self.assertIn(audit["normalization_path"], receipt["artifact_hashes"])
            self.assertEqual(verify_publication(root), receipt)
            candidate.write_text("changed after delivery")
            with self.assertRaises(PipelineError):
                verify_publication(root)


if __name__ == "__main__":
    unittest.main()
