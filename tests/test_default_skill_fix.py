from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from workbench.cli import build_parser
from workbench.config import NextEchoConfig
from workbench.contracts import PipelineError
from workbench.headless_cli import cmd_run
from workbench.io_utils import write_json
from workbench.nemotron_candidate import validate_assets
from workbench.speaker_service import build_run_speakers
from tests.test_publication import factory_item


class DefaultSkillFixTests(unittest.TestCase):
    def invoke(self, approved, *, profile='fast'):
        cfg = NextEchoConfig({'agent': {'default_profile': profile, 'approved_trial_models': approved},
                              'profiles': {'fast': {'polish': 'qwen35-4b'}}})
        args = build_parser().parse_args(['run', 'https://example.invalid/episode', '--detach', '--output', 'json'])
        store = MagicMock()
        store.submit.return_value = ({'schema_version': 1, 'job_id': 'queued-job', 'status': 'queued'}, False)
        store.get.return_value = store.submit.return_value[0]
        output = io.StringIO()
        with patch('workbench.headless_cli.bound_config', return_value=cfg), \
             patch('workbench.headless_cli.JobStore', return_value=store), contextlib.redirect_stdout(output):
            code = cmd_run(args)
        return code, json.loads(output.getvalue()), store, cfg

    def test_url_only_default_stays_blocked_without_opt_in(self):
        code, result, store, _ = self.invoke([])
        self.assertEqual((code, result['error_code']), (5, 'MODEL_CANDIDATE'))
        store.submit.assert_not_called()

    def test_asr_approval_does_not_silently_approve_polish(self):
        code, result, store, _ = self.invoke(['sensevoice-int8'])
        self.assertEqual((code, result['error_code']), (5, 'MODEL_CANDIDATE'))
        store.submit.assert_not_called()

    def test_approved_local_pair_runs_without_profile_flag_and_freezes_consent(self):
        approved = ['sensevoice-int8', 'qwen35-4b']
        code, result, store, cfg = self.invoke(approved)
        self.assertEqual((code, result['status']), (0, 'queued'))
        plan = store.submit.call_args.args[0]
        self.assertEqual((plan['request']['profile'], plan['asr_profile'], plan['polish_profile']),
                         ('fast', 'sensevoice-int8', 'qwen35-4b'))
        self.assertEqual(plan['effective']['agent']['approved_trial_models'], approved)
        self.assertNotIn('models', cfg)  # No candidate is promoted to ready.

    def test_approval_does_not_apply_to_another_default_model(self):
        code, result, store, _ = self.invoke(['sensevoice-int8', 'qwen35-4b'], profile='balanced')
        self.assertEqual((code, result['error_code']), (5, 'MODEL_CANDIDATE'))
        store.submit.assert_not_called()

    def test_wildcard_or_malformed_opt_in_is_rejected(self):
        for approved in ('all', ['*'], [True]):
            with self.subTest(approved=approved):
                code, result, store, _ = self.invoke(approved)
                self.assertEqual((code, result['error_code']), (5, 'CONFIG_ERROR'))
                store.submit.assert_not_called()

    def test_absent_and_corrupt_native_ledgers_remain_distinct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); ledger = root / 'ledger.json'
            with self.assertRaises(PipelineError) as missing:
                validate_assets(ledger, root)
            self.assertEqual(missing.exception.code, 'MODEL_MISSING')
            ledger.write_text('{broken JSON')
            with self.assertRaises(PipelineError) as corrupt:
                validate_assets(ledger, root)
            self.assertEqual(corrupt.exception.code, 'MODEL_INVALID')

    def test_missing_optional_speakers_preserves_asr_while_required_and_corruption_block(self):
        for mode, error, should_raise in [('auto', 'MODEL_MISSING', False),
                                           ('required', 'MODEL_MISSING', True),
                                           ('auto', 'MODEL_INVALID', True)]:
            with self.subTest(mode=mode, error=error), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                item = factory_item(root, 'one'); item['char_count'] = 10
                write_json(root / 'manifest.json', {'results': [item]})
                source = Path(item['text_path']).read_bytes()
                with patch('workbench.speaker_service.build_item_speakers',
                           side_effect=PipelineError(error, 'fixture', blocked=True)):
                    if should_raise:
                        with self.assertRaises(PipelineError) as raised:
                            build_run_speakers(root, mode=mode)
                        self.assertEqual(raised.exception.code, error)
                    else:
                        self.assertEqual(build_run_speakers(root, mode=mode)['status'], 'partial')
                        row = json.loads((root / 'manifest.json').read_text())['results'][0]
                        self.assertEqual(row['speaker_status'], 'partial')
                        self.assertEqual(row['speaker_error']['code'], 'MODEL_MISSING')
                self.assertEqual(Path(item['text_path']).read_bytes(), source)


if __name__ == '__main__':
    unittest.main()
