from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from workbench import postprocess as P
from workbench.enhancement import default_glossary_rules
from workbench.io_utils import file_hash, write_json
from workbench.llm_providers import CONTEXT, GenerationContext
from workbench.reading_view import clean_boundaries, render_reading, chapters_from_context
from workbench.resume import resume_plan
from workbench.contracts import PipelineError
from workbench.asr_cache import NATIVE_CODE, relocated_native_identity


class ReadingBoundaryTests(unittest.TestCase):
    def row(self, id, text, speaker, start):
        return {"id": id, "text": text, "speaker": speaker, "speaker_id": "" if speaker == "归属待核对" else speaker,
                "speaker_uuid": "", "start_ms": start, "end_ms": start + 200}

    def test_reading_only_removes_ack_and_retains_meaningful_negation_and_boundary(self):
        source = [self.row("a", "费用", "说话人 1", 0), self.row("b", "嗯，对。", "归属待核对", 201),
                  self.row("c", "不能", "归属待核对", 410), self.row("d", "超过36万元。", "说话人 1", 620)]
        original = copy.deepcopy(source)
        rows, audit = clean_boundaries(source)
        self.assertEqual(source, original)
        self.assertEqual("".join(r["text"] for r in rows), "费用不能超过36万元。")
        self.assertEqual(audit["omitted_backchannels"], 1)
        self.assertEqual(audit["neighbor_joins"], 1)
        body, audit = render_reading(source, {})
        self.assertIn("费用不能超过36万元。", body)
        self.assertNotIn("归属待核对", body)
        self.assertEqual(audit["accounted_segment_ids"], ["a", "b", "c", "d"])

    def test_substantive_short_reply_is_not_deleted_and_no_distant_identity_guess(self):
        source = [self.row("a", "不能。", "归属待核对", 0), self.row("b", "是", "归属待核对", 500),
                  self.row("c", "这一项需要确认。", "说话人 2", 10000)]
        rows, audit = clean_boundaries(source)
        self.assertEqual(len(rows), 3)
        self.assertEqual(audit["omitted_backchannels"], 0)
        self.assertEqual(audit["neighbor_joins"], 0)

    def test_full_chapters_do_not_mix_timestamped_glossary_or_truncate_at_16(self):
        chapters = [{"start_ms": n * 1000, "text": f"00:{n:02d} 主题{n}"} for n in range(1, 21)]
        context = {"shownotes": {"sections": chapters + [{"start_ms": 500, "text": "术语定义"}]}}
        self.assertEqual(len(chapters_from_context(context)), 20)
        body, audit = render_reading([self.row("a", "完整结尾", "说话人 1", 21000)], context)
        self.assertEqual(audit["rendered_chapters"], 20)
        self.assertIn('id="item-1-chapter-19"', body)
        self.assertNotIn("术语定义", body)


class NativeCacheInstallationTests(unittest.TestCase):
    def test_equal_code_can_rekey_installation_paths_but_changed_code_cannot(self):
        with tempfile.TemporaryDirectory(prefix='cache spaces ') as d:
            root = Path(d); old = root/'old/workbench'; new = root/'new/workbench'
            old.mkdir(parents=True); new.mkdir(parents=True)
            for name in NATIVE_CODE:
                (old/name).write_text(name); (new/name).write_text(name)
            identity = {str(new/name):file_hash(new/name) for name in NATIVE_CODE}
            identity['model-weights'] = 'fixed-weights'
            candidate = relocated_native_identity({'cache_implementation_root':str(old)}, root, identity, '/python', new)
            self.assertEqual(candidate[0]['model-weights'], 'fixed-weights')
            self.assertEqual(candidate[0][str(old/'local_asr.py')], file_hash(new/'local_asr.py'))
            (old/'local_asr.py').write_text('changed implementation')
            self.assertIsNone(relocated_native_identity({'cache_implementation_root':str(old)}, root, identity, '/python', new))


class SegmentRepairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.source = [P.Segment("a", "金额36万元，不能增加。", 0, 1000, "A"),
                       P.Segment("b", "这是原始观点。", 1000, 2000, "B")]

    def old_candidate(self, *, wrong_time=False, forged_number=False):
        path = self.folder / "candidates" / "old.json"
        rows = [{"id": "a", "polished_text": self.source[0].text.replace("36", "360")},
                {"id": "b", "polished_text": "这是原始观点。"}]
        if wrong_time:
            rows[0] = {"id": "a", "polished_text": self.source[0].text, "end_ms": 999}
        write_json(path, {"model": "fixture", "input_segments": [P.segment_to_dict(s) for s in self.source],
                          "candidate": {"segments": rows}, "validation": "accepted-protected-fields" if forged_number else "rejected"})
        record = {"model": "fixture", "candidate_path": str(path), "sha256": file_hash(path),
                  "path_protocol": P.CANDIDATE_PATH_PROTOCOL, "candidate_relative_path": "candidates/old.json",
                  "validation": "accepted-protected-fields" if forged_number else "rejected"}
        write_json(self.folder / "chunk_000.json", {"candidate_attempts": [record]})
        return path

    def repair(self, callback):
        token = CONTEXT.set(GenerationContext({"repair_candidates": True, "preserve_standalone_acknowledgements": False,
                                               "context_tokens": 8192, "max_output_tokens": 2048}))
        try:
            with patch.object(P, "call_json", side_effect=callback):
                return P.polish_segments(self.source, rules=default_glossary_rules(), models=["fixture"],
                    profile="balanced", style="", endpoint="http://127.0.0.1:1", checkpoint_dir=self.folder)
        finally:
            CONTEXT.reset(token)

    def test_revalidates_rejected_chunk_by_sentence_and_repairs_only_bad_row(self):
        self.old_candidate()
        calls = []
        def model(**kw):
            data = json.loads(kw["user"].split("\n", 1)[1]); calls.extend(r["id"] for r in data["segments"])
            self.assertNotIn("end_ms", kw["schema"]["properties"]["segments"]["items"]["properties"])
            return {"segments": [{"id": r["id"], "polished_text": r["text"]} for r in data["segments"]]}, "fixture", []
        output, audit = self.repair(model)
        self.assertEqual(calls, ["a"])
        self.assertEqual(output, self.source)
        self.assertEqual(audit["reused_segments"], 1)
        self.assertEqual(audit["unprocessed_segment_ids"], [])
        self.assertTrue(audit["processing_complete"])
        self.assertTrue(audit["protected_fields_pass"])

    def test_changed_metadata_is_rejected_not_silently_stripped(self):
        self.old_candidate(wrong_time=True)
        calls = []
        def model(**kw):
            data = json.loads(kw["user"].split("\n", 1)[1]); calls.extend(r["id"] for r in data["segments"])
            return {"segments": [{"id": "a", "polished_text": self.source[0].text}]}, "fixture", []
        output, audit = self.repair(model)
        self.assertEqual(calls, ["a"])
        self.assertEqual(output[0].end_ms, 1000)
        self.assertTrue(any("end_ms" in r["reason"] for r in audit["guard_rejections"]))

    def test_forged_accepted_candidate_cannot_bypass_guard_and_final_source_is_safe(self):
        self.old_candidate(forged_number=True)
        def bad(**kw):
            return {"segments": [{"id": "a", "polished_text": self.source[0].text.replace("36", "360")}]}, "fixture", []
        output, audit = self.repair(bad)
        self.assertEqual(output, self.source)
        self.assertEqual(audit["retained_source_segments"], ["a"])
        self.assertEqual(audit["status"], "partial")
        self.assertTrue(audit["protected_fields_pass"])

    def test_tampered_candidate_does_not_reuse_unverified_rows(self):
        path = self.old_candidate()
        path.write_text("{}")
        calls = []
        def model(**kw):
            data = json.loads(kw["user"].split("\n", 1)[1]); calls.extend(r["id"] for r in data["segments"])
            return {"segments": [{"id": r["id"], "polished_text": r["text"]} for r in data["segments"]]}, "fixture", []
        output, audit = self.repair(model)
        self.assertCountEqual(calls, ["a", "b"])
        self.assertEqual(audit["reused_segments"], 0)
        self.assertEqual(output, self.source)

    def test_execution_failure_propagates_instead_of_successful_fallback(self):
        with self.assertRaises(PipelineError):
            self.repair(lambda **kw: (_ for _ in ()).throw(PipelineError("RESOURCE_LIMIT", "fixture")))

    def test_interrupted_repair_revalidates_checkpoint_and_does_not_repeat_finished_sentence(self):
        self.source = [P.Segment('a', '原始观点。' * 80, 0, 1000, 'A'),
                       P.Segment('b', '后续内容。' * 80, 1000, 2000, 'B')]
        calls = []
        def model(**kw):
            data = json.loads(kw['user'].split('\n', 1)[1]); ids = [r['id'] for r in data['segments']]
            calls.extend(ids)
            if ids == ['b']:
                raise PipelineError('CANCELLED', 'fixture interruption')
            return {'segments': [{'id': r['id'], 'polished_text': r['text']} for r in data['segments']]}, 'fixture', []
        with self.assertRaises(PipelineError):
            self.repair(model)
        def finish(**kw):
            data = json.loads(kw['user'].split('\n', 1)[1]); calls.extend(r['id'] for r in data['segments'])
            return {'segments': [{'id': r['id'], 'polished_text': r['text']} for r in data['segments']]}, 'fixture', []
        output, audit = self.repair(finish)
        self.assertEqual(calls, ['a', 'b', 'b'])
        self.assertEqual(audit['reused_segments'], 1)
        self.assertEqual(output, self.source)


if __name__ == "__main__":
    unittest.main()
