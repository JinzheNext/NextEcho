import html
import json
import re
import tempfile
import unittest
from pathlib import Path

from workbench.review import text_change, write_review


class ReviewDiffTests(unittest.TestCase):
    def test_chinese_diffs_reconstruct_both_real_texts(self):
        before, after = "我我觉得预算36万元不能增加", "我觉得，预算36万元不能增加。"
        d = text_change(before, after)
        def strip(s):
            return html.unescape(re.sub('<[^>]*>', '', s))
        self.assertEqual(strip(d['left']), before)
        self.assertEqual(strip(d['right']), after)
        self.assertIn('<del', d['left'])
        self.assertIn('<ins', d['right'])
        self.assertEqual(d['kind'], 'wording')
        self.assertEqual(text_change('不能增加', '不能增加。')['kind'], 'punctuation')
        self.assertEqual(text_change('没改', '没改')['kind'], 'unchanged')

    def test_hostile_transcript_never_becomes_markup_or_script(self):
        hostile = '</script><img src=x onerror=alert(1)> & "测试"'
        record = {'id': 'a', 'suite': 'business60', 'original': hostile,
                  'models': {'4B': {'text': hostile + '。', 'fallback': False}}}
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'review.html'
            write_review(p, [record])
            source = p.read_text()
        embedded = source.split('<script id="review-data" type="application/json">')[1].split('</script>')[0]
        self.assertNotIn('<img', embedded)
        self.assertEqual(json.loads(embedded)['records'][0]['original'], hostile)
        self.assertIn('&lt;img', json.loads(embedded)['records'][0]['models']['4B']['diff']['left'])


if __name__ == '__main__':
    unittest.main()
