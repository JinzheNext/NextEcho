from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from workbench import postprocess as P
from workbench.enhancement import default_glossary_rules
from workbench.fidelity import canonical_quantities, high_risk_edits, text_patch_evidence


class FidelityTests(unittest.TestCase):
    def test_rejects_polarity_modality_negation_scope_and_financial_roles(self):
        cases = [
            ("今年亏损100万元。", "今年盈利100万元。"),
            ("这个方案可能有效。", "这个方案必然有效。"),
            ("我不赞成这个方案。", "我赞成这个方案。"),
            ("我不赞成甲，但赞成乙。", "我赞成甲，但不赞成乙。"),
            ("我不赞成甲，但赞成乙。", "我不赞成乙，但赞成甲。"),
            ("收入100元，利润20元。", "利润100元，收入20元。"),
            ("收入100元，利润20元。", "收入20元，利润100元。"),
            ("预算三四万。", "预算三十万。"),
            ("预算约三万元。", "预算三万元。"),
            ("至少3个人。", "至多3个人。"),
            ("使用OpenAI模型。", "使用OpenAl模型。"),
            ("版本2.0。", "版本2。"),
            ("邮箱一万@example.com。", "邮箱10000@example.com。"),
        ]
        for before, after in cases:
            with self.subTest(before=before, after=after):
                self.assertTrue(high_risk_edits(before, after))
                with self.assertRaises(ValueError):
                    P.ensure_preserved(before, after, default_glossary_rules())

    def test_certifies_supported_equivalent_quantity_forms_and_punctuation(self):
        cases = [
            ("预算一万元。", "预算1万元。"),
            ("预算一万元。", "预算10000元。"),
            ("金额一百零二元。", "金额102元。"),
            ("一万零三元。", "10003元。"),
            ("负一万元。", "-10000元。"),
            ("二〇二六年。", "2026年。"),
            ("占比百分之三。", "占比3%。"),
            ("金额1.50元。", "金额一点五元。"),
            ("金额一点五万元。", "金额15000元。"),
            ("我我不赞成甲但赞成乙。", "我不赞成甲，但赞成乙。"),
            ("我们，呃，不能超支。", "我们不能超支。"),
            ("OK2023年。", "OK，2023 年。"),
            ("使用Social功能。", "使用Social.功能。"),
            ("我們沒有改變利潤。", "我们没有改变利润。"),
        ]
        for before, after in cases:
            with self.subTest(before=before, after=after):
                self.assertEqual(high_risk_edits(before, after), [])
                P.ensure_preserved(before, after, default_glossary_rules())

    def test_ambiguous_elision_and_long_identifiers_remain_literal(self):
        for before, after in [("金额一百二元。", "金额102元。"), ("金额一百二元。", "金额120元。"),
                              ("金额一万二元。", "金额10002元。")]:
            with self.assertRaises(ValueError):
                P.ensure_preserved(before, after, default_glossary_rules())
        identifier = "123456789012345678901234567890123456789012345678901234567890123456789"
        before = "编号" + identifier
        P.ensure_preserved(before, before, default_glossary_rules())
        with self.assertRaises(ValueError):
            P.ensure_preserved(before, before[:-1] + "8", default_glossary_rules())
        literal = "参考https://example.test/一万元和`一万元`。"
        self.assertEqual(canonical_quantities(literal), (literal, []))

    def test_patch_offsets_reconstruct_the_exact_adopted_text(self):
        before, after = "金额100元，呃，我们继续。", "金额100元，我们继续！"
        evidence = text_patch_evidence(before, after, segment_id="s", layer="adopted")
        rebuilt = before
        for edit in reversed(evidence["edits"]):
            self.assertEqual(before[edit["source_start"]:edit["source_end"]], edit["removed"])
            rebuilt = rebuilt[:edit["source_start"]] + edit["replacement"] + rebuilt[edit["source_end"]:]
        self.assertEqual(rebuilt, after)
        self.assertEqual(evidence["source_sha256"], hashlib.sha256(before.encode()).hexdigest())
        self.assertEqual(evidence["semantic_review"], "pending")

    def test_rejected_model_candidates_are_saved_and_fallback_is_original_input(self):
        source = P.Segment("s", "金额100元，呃，我们不能超支。", 100, 3000, "A")
        response = {"segments": [{"id": "s", "polished_text": "金额100元，我们可以超支。"}]}
        with tempfile.TemporaryDirectory() as directory, patch.object(P, "call_json", return_value=(response, "probe", [])) as call:
            output, audit = P.polish_segments([source], rules=default_glossary_rules(), models=["probe"],
                profile="balanced", style="", endpoint="http://127.0.0.1:1", checkpoint_dir=Path(directory))
            self.assertEqual(output, [source])
            self.assertEqual(call.call_count, 2)
            self.assertEqual(audit["status"], "partial")
            self.assertEqual(audit["fallback_input_layer"], "original-segment-input")
            self.assertEqual(len(audit["candidate_attempts"]), 2)
            for record in audit["candidate_attempts"]:
                path = Path(record["candidate_path"])
                saved = json.loads(path.read_text())
                self.assertEqual(saved["candidate"], response)
                self.assertEqual(saved["validation"], "rejected")
                self.assertEqual(saved["input_segments"][0]["text"], "金额100元，我们不能超支。")
                self.assertEqual(record["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_candidate_evidence_and_guard_revision_are_required_for_cache_reuse(self):
        source = P.Segment("s", "金额100元，呃，我们继续。", 0, 3000, "A")
        response = {"segments": [{"id": "s", "polished_text": "金额100元，我们继续！"}]}
        with tempfile.TemporaryDirectory() as directory, patch.object(P, "call_json", return_value=(response, "probe", [])) as call:
            def run():
                return P.polish_segments([source], rules=default_glossary_rules(), models=["probe"],
                    profile="balanced", style="", endpoint="http://127.0.0.1:1", checkpoint_dir=Path(directory))
            first, audit = run()
            self.assertEqual(first[0].text, "金额100元，我们继续！")
            self.assertTrue(audit["protected_fields_pass"])
            self.assertEqual(audit["semantic_review"], "pending")
            self.assertTrue(audit["adopted_patches"][0]["edits"])
            _, cached = run()
            self.assertEqual(call.call_count, 1)
            self.assertEqual(cached["candidate_attempts"], audit["candidate_attempts"])
            Path(audit["candidate_attempts"][0]["candidate_path"]).unlink()
            run()
            self.assertEqual(call.call_count, 2)
            with patch.object(P, "fidelity_implementation", return_value={"validator_version": "changed-for-test"}):
                run()
            self.assertEqual(call.call_count, 3)

    def test_uncertain_scope_edit_is_visible_review_candidate_not_silent_ready(self):
        source = P.Segment("s", "我可能讲的是李纪刚。", 0, 3000, "A")
        response = {"segments": [{"id": "s", "polished_text": "我可能讲的是李继刚。"}]}
        with tempfile.TemporaryDirectory() as directory, patch.object(P, "call_json", return_value=(response, "probe", [])) as call:
            def run():
                return P.polish_segments([source], rules=default_glossary_rules(), models=["probe"],
                    profile="balanced", style="", endpoint="http://127.0.0.1:1", checkpoint_dir=Path(directory))
            output, audit = run()
            self.assertEqual(output[0].text, response["segments"][0]["polished_text"])
            self.assertEqual(audit["status"], "partial")
            self.assertTrue(audit["review_required"])
            self.assertEqual(audit["review_segments"][0]["segment_id"], "s")
            self.assertEqual(audit["original_fallback_chunks"], [])
            _, cached = run()
            self.assertEqual(call.call_count, 1)
            self.assertEqual(cached["status"], "partial")
            self.assertTrue(cached["review_required"])
            raw = {"transcription": [P.segment_to_transcription(source)]}
            P.write_enhanced_artifacts(Path(directory), raw, output, default_glossary_rules(), audit, "balanced")
            written = json.loads((Path(directory) / "transcript.enhanced.json").read_text())
            self.assertEqual(written["transcription"], raw["transcription"])
            self.assertTrue(written["enhancement"]["ai"]["applied"])
            self.assertTrue(written["enhancement"]["ai"]["review_required"])
            self.assertEqual(written["enhancement"]["postprocess"]["status"], "partial")


if __name__ == "__main__":
    unittest.main()
