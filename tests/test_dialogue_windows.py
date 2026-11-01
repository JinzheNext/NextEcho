import unittest

from workbench.dialogue_worker import exclusive_speaker_windows


class DialogueWindowTests(unittest.TestCase):
    def test_overlapping_voices_are_not_duplicated_or_falsely_owned(self):
        source = [{"start": 0, "end": 10, "speaker": "对话人1", "acoustic_label": "s0"},
                  {"start": 4, "end": 6, "speaker": "对话人2", "acoustic_label": "s1"}]
        rows = exclusive_speaker_windows(source)
        self.assertEqual([(r['start'], r['end']) for r in rows], [(0, 4), (4, 6), (6, 10)])
        self.assertEqual([r['overlap'] for r in rows], [False, True, False])
        self.assertEqual(sum(r['end'] - r['start'] for r in rows), 10)
        self.assertIn('对话人1＋对话人2', rows[1]['speaker'])
        self.assertEqual(source[0]['end'], 10)

    def test_empty_and_non_speech_gaps_are_not_invented(self):
        self.assertEqual(exclusive_speaker_windows([]), [])
        rows = exclusive_speaker_windows([{'start': 1, 'end': 2, 'speaker': '对话人1', 'acoustic_label': 'a'},
                                          {'start': 5, 'end': 8, 'speaker': '对话人1', 'acoustic_label': 'a'}])
        self.assertEqual([(r['start'], r['end']) for r in rows], [(1, 2), (5, 8)])
