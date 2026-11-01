from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app import app


class ReviewArtifactTests(unittest.TestCase):
    def test_active_sources_and_generated_review_cannot_execute_on_app_origin(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run = root / 'run_test'
            run.mkdir()
            for name in ('source.html', 'review.html', 'source.svg', 'source.html.gz'):
                (run / name).write_text('<script>fetch("/api/jobs")</script>')
                with patch('app.RUNS_ROOT', root):
                    response = app.test_client().get('/artifacts/run_test/' + name + '?download=0')
                    self.addCleanup(response.close)
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.headers['Content-Disposition'], 'attachment')
                    self.assertIn('sandbox', response.headers['Content-Security-Policy'])
                    self.assertEqual(response.headers['X-Content-Type-Options'], 'nosniff')
                    self.assertEqual(response.headers['Cache-Control'], 'no-store')

    def test_plain_text_and_media_keep_inline_range_reading_but_external_symlink_is_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run = root / 'run_test'
            run.mkdir()
            (run / 'audio.wav').write_bytes(b'0123456789')
            outside = root / 'outside.txt'
            outside.write_text('outside data')
            (run / 'escape.txt').symlink_to(outside)
            with patch('app.RUNS_ROOT', root):
                client = app.test_client()
                response = client.get('/artifacts/run_test/audio.wav', headers={'Range': 'bytes=2-5'})
                self.addCleanup(response.close)
                self.assertEqual(response.status_code, 206)
                self.assertEqual(response.data, b'2345')
                self.assertNotIn('attachment', response.headers.get('Content-Disposition', ''))
                blocked = client.get('/artifacts/run_test/escape.txt')
                self.assertEqual(blocked.status_code, 404)
                self.assertNotIn(b'outside data', blocked.data)
