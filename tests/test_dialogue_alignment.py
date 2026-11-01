import unittest

from workbench.dialogue_alignment import align_dialogue_segments


class DialogueAlignmentTests(unittest.TestCase):
    def test_native_anchors_split_voices_and_keep_exact_source_text(self):
        source = [{'text': '你好。可以！', 'start_ms': 0, 'end_ms': 2000}]
        tokens = [{'text': c, 'start_ms': i * 300, 'end_ms': i * 300 + 100} for i, c in enumerate('你好。可以！')]
        turns = [{'start': 0, 'end': .8, 'speaker': '对话人1'}, {'start': .8, 'end': 2, 'speaker': '对话人2'}]
        rows, audit = align_dialogue_segments(source, tokens, turns, prefix='test')
        self.assertEqual([r['text'] for r in rows], ['你好。', '可以！'])
        self.assertEqual([r['speaker'] for r in rows], ['对话人1', '对话人2'])
        self.assertFalse(audit['word_timestamps_generated'])

    def test_unmatched_text_across_voice_change_is_explicitly_uncertain(self):
        source = [{'text': '甲未知乙', 'start_ms': 0, 'end_ms': 4000}]
        tokens = [{'text': '甲', 'start_ms': 0, 'end_ms': 100}, {'text': '乙', 'start_ms': 3000, 'end_ms': 3100}]
        turns = [{'start': 0, 'end': 1, 'speaker': '对话人1'}, {'start': 2, 'end': 4, 'speaker': '对话人2'}]
        rows, _ = align_dialogue_segments(source, tokens, turns, prefix='test')
        self.assertEqual(''.join(r['text'] for r in rows), '甲未知乙')
        self.assertTrue(any(r['speaker'] == '归属待核对' and r['text'] == '未知' for r in rows))

    def test_explicit_zero_width_alignment_is_not_treated_as_onset_only(self):
        rows, _ = align_dialogue_segments(
            [{'id': 's', 'text': '甲', 'start_ms': 0, 'end_ms': 1000}],
            [{'text': '甲', 'start_ms': 400, 'end_ms': 400}],
            [{'start': 0, 'end': 1, 'speaker': 'A'}], prefix='test',
        )
        evidence = rows[0]['alignment_spans'][0]
        self.assertEqual(rows[0]['speaker'], '归属待核对')
        self.assertEqual(evidence['assignment_reason'], 'missing_duration')
        self.assertEqual(evidence['native_start_ms'], evidence['native_end_ms'])

    def test_missing_duration_still_uses_a_half_open_native_onset(self):
        rows, _ = align_dialogue_segments(
            [{'id': 's', 'text': '甲', 'start_ms': 0, 'end_ms': 1000}],
            [{'text': '甲', 'start_ms': 400, 'end_ms': None}],
            [{'start': 0, 'end': .5, 'speaker': 'A'}], prefix='test',
        )
        self.assertEqual(rows[0]['speaker'], 'A')
        self.assertEqual(rows[0]['alignment_spans'][0]['assignment_reason'], 'native_onset')
