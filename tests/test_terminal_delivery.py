"""Queries validate delivery while preserving immutable execution history."""
from __future__ import annotations

import contextlib
from argparse import Namespace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from workbench.delivery import assemble,write_receipt
from workbench.headless_cli import cmd_jobs,emit
from workbench.job_store import JobStore
from workbench.config import load_config
from workbench.contracts import PipelineRequest, plan_request


class TerminalDeliveryTests(unittest.TestCase):
    def finished(self,root):
        store=JobStore(root/'registry')
        plan=plan_request(PipelineRequest(('https://example.org/fixture.wav',),postprocess='off'),load_config()).to_dict()
        job,_=store.submit(plan,output_dir=root/'run',key='same')
        run=Path(job['run_dir'])
        text=run/'transcript.txt'
        text.write_text('合成逐字稿，用于文件完整性测试。')
        results=[{'title':'合成节目','status':'ready','text_path':str(text)}]
        primary=assemble(run,results)
        receipt=write_receipt(run,job_id=job['job_id'],status='ready',results=results,plan=plan)
        saved=store.update(job['job_id'],**{k:v for k,v in receipt.items() if k not in {'job_id','plan'}})
        return store,saved,primary,text

    def test_modified_primary_blocks_query_and_same_key_without_rewriting_history(self):
        with tempfile.TemporaryDirectory() as d:
            store,saved,primary,_=self.finished(Path(d))
            self.assertEqual(store.get(saved['job_id'])['delivery_integrity'],'verified')
            original=primary.read_bytes()
            primary.write_text('changed after completion')
            checked=store.get(saved['job_id'])
            self.assertEqual(checked['status'],'blocked')
            self.assertEqual(checked['execution_status'],'ready')
            self.assertEqual(checked['primary_file'],'')
            self.assertEqual(checked['error_code'],'CACHE_MISMATCH')
            repeated,created=store.submit(saved['plan'],output_dir=Path(d)/'run',key='same')
            self.assertFalse(created)
            self.assertEqual(repeated['job_id'],saved['job_id'])
            self.assertEqual(repeated['status'],'blocked')
            self.assertEqual(store.get(saved['job_id'],reconcile=False),saved)
            self.assertEqual(json.loads(Path(saved['receipt_path']).read_text())['status'],'ready')
            stdout,stderr=io.StringIO(),io.StringIO()
            with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
                code=emit(checked,'paths')
            self.assertEqual(code,5)
            self.assertEqual(stdout.getvalue(),'')
            primary.write_bytes(original)
            self.assertEqual(store.get(saved['job_id'])['status'],'ready')

    def test_changed_item_missing_receipt_and_malformed_receipt_are_rejected(self):
        for corruption in ['item','receipt-missing','receipt-not-object','receipt-ledger']:
            with self.subTest(corruption=corruption),tempfile.TemporaryDirectory() as d:
                store,saved,_,text=self.finished(Path(d))
                receipt=Path(saved['receipt_path'])
                if corruption=='item':
                    text.write_text('modified supporting artifact')
                elif corruption=='receipt-missing':
                    receipt.unlink()
                elif corruption=='receipt-not-object':
                    receipt.write_text('[]')
                else:
                    payload=json.loads(receipt.read_text())
                    payload['artifact_hashes']={}
                    receipt.write_text(json.dumps(payload))
                checked=store.get(saved['job_id'])
                self.assertEqual(checked['status'],'blocked')
                self.assertEqual(checked['delivery_integrity'],'invalid')
                self.assertEqual(store.get(saved['job_id'],reconcile=False),saved)

    def test_invalid_delivery_can_resume_with_original_terminal_history_intact(self):
        with tempfile.TemporaryDirectory() as d:
            store,saved,primary,_=self.finished(Path(d))
            receipt_bytes=Path(saved['receipt_path']).read_bytes()
            primary.write_text('invalidated delivery')
            stdout=io.StringIO()
            with patch('workbench.headless_cli.JobStore',return_value=store), \
                    patch.object(store,'spawn') as spawn,contextlib.redirect_stdout(stdout):
                code=cmd_jobs(Namespace(action='resume',job_id=saved['job_id'],output='json'))
            self.assertEqual(code,0)
            resumed=json.loads(stdout.getvalue())
            self.assertNotEqual(resumed['job_id'],saved['job_id'])
            spawn.assert_called_once_with(resumed['job_id'])
            child=store.get(resumed['job_id'],reconcile=False)
            self.assertEqual(child['parent_job_id'],saved['job_id'])
            self.assertEqual(child['attempt'],2)
            self.assertEqual(Path(saved['receipt_path']).read_bytes(),receipt_bytes)
            self.assertEqual(store.get(saved['job_id'],reconcile=False),saved)

    def test_malformed_old_plan_is_not_silently_resumed(self):
        with tempfile.TemporaryDirectory() as d:
            store=JobStore(Path(d)/'registry')
            job,_=store.submit({'request':{}})
            saved=store.update(job['job_id'],status='failed')
            stdout=io.StringIO()
            with patch('workbench.headless_cli.JobStore',return_value=store), \
                    patch.object(store,'spawn') as spawn,contextlib.redirect_stdout(stdout):
                code=cmd_jobs(Namespace(action='resume',job_id=saved['job_id'],output='json'))
            self.assertEqual(code,5)
            self.assertEqual(json.loads(stdout.getvalue())['error_code'],'PROTOCOL_ERROR')
            spawn.assert_not_called()
            self.assertEqual(store.get(saved['job_id'],reconcile=False),saved)


if __name__=='__main__':
    unittest.main()
