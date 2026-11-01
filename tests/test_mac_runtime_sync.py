"""Exercise real rsync only inside a disposable project/support tree."""
from __future__ import annotations

import os
from pathlib import Path
import plistlib
import shlex
import shutil
import subprocess
import tempfile
import unittest

try:
    import fcntl
except ImportError:
    fcntl = None


@unittest.skipUnless(fcntl is not None and shutil.which('rsync'), 'POSIX + rsync')
class MacRuntimeSyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='nextecho-mac-sync-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root/'project'
        self.support = self.root/'support'
        self.state = self.support/'runtime'
        self.app = self.support/'app'
        self.state.mkdir(parents=True)
        scripts = self.project/'scripts'
        scripts.mkdir(parents=True)
        source = Path(__file__).resolve().parents[1]/'scripts'
        for name in ['sync_mac_runtime.sh', 'mac_launch_agent.sh']:
            text = (source/name).read_text().replace(
                'SUPPORT_DIR="$HOME/Library/Application Support/NextEcho"',
                'SUPPORT_DIR='+shlex.quote(str(self.support)))
            text = text.replace('PLIST_PATH="$HOME/Library/LaunchAgents/$LABEL.plist"',
                                'PLIST_PATH='+shlex.quote(str(self.root/'agent.plist')))
            text = text.replace('LOG_DIR="$HOME/Library/Logs/NextEcho"',
                                'LOG_DIR='+shlex.quote(str(self.root/'logs')))
            text = text.replace('"$HOME/Library/LaunchAgents"', shlex.quote(str(self.root/'agents')))
            (scripts/name).write_text(text)
        (self.project/'app.py').write_text('# synthetic application\n')
        self.env = {**os.environ, 'NEXTECHO_RUNTIME_DIR': str(self.state)}

    def run_script(self, name='sync_mac_runtime.sh', *args):
        return subprocess.run(['bash',str(self.project/'scripts'/name),*args],
                              env=self.env,text=True,capture_output=True,timeout=15)

    def test_sync_preserves_state_and_live_lock_and_removes_only_stale_code(self):
        database = self.state/'jobs.sqlite3'
        database.write_bytes(b'synthetic state sentinel')
        binding = self.state/'binding.json'
        binding.write_text('{"synthetic":true}')
        lock = self.state/'heavy.lock'
        self.app.mkdir()
        (self.app/'stale-code.py').write_text('# removable code copy')
        # A source copied from the old combined directory must not copy state.
        (self.project/'jobs.sqlite3').write_bytes(b'do not copy this database')
        with lock.open('a+') as held:
            fcntl.flock(held, fcntl.LOCK_EX)
            inode = lock.stat().st_ino
            result = self.run_script()
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(database.read_bytes(),b'synthetic state sentinel')
            self.assertEqual(binding.read_text(),'{"synthetic":true}')
            self.assertEqual(lock.stat().st_ino,inode)
            with lock.open('a+') as contender:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.assertFalse((self.app/'stale-code.py').exists())
        self.assertTrue((self.app/'app.py').is_file())
        self.assertFalse((self.app/'jobs.sqlite3').exists())
        self.assertEqual(Path(result.stdout.strip()),self.app)
        repeated=subprocess.run(['bash',str(self.app/'scripts/sync_mac_runtime.sh')],
                                env=self.env,text=True,capture_output=True,timeout=15)
        self.assertEqual(repeated.returncode,0,repeated.stderr)
        self.assertEqual(database.read_bytes(),b'synthetic state sentinel')

    def test_alias_or_nested_state_destination_is_rejected_before_sync(self):
        sentinel=self.state/'jobs.sqlite3'
        sentinel.write_bytes(b'keep')
        self.app.symlink_to(self.state,target_is_directory=True)
        result=self.run_script()
        self.assertNotEqual(result.returncode,0)
        self.assertIn('directories overlap',result.stderr)
        self.assertEqual(sentinel.read_bytes(),b'keep')
        self.app.unlink()
        self.env['NEXTECHO_RUNTIME_DIR']=str(self.app/'state')
        self.assertNotEqual(self.run_script().returncode,0)
        self.assertFalse(self.app.exists())

    def test_launcher_syncs_before_environment_check_and_uses_app_directory(self):
        python=self.project/'.venv/bin/python'
        python.parent.mkdir(parents=True)
        python.write_text('#!/bin/sh\nexit 0\n')
        python.chmod(0o755)
        (self.project/'.venv/lib/python3.12/site-packages').mkdir(parents=True)
        bin_dir=self.root/'bin'
        bin_dir.mkdir()
        launchctl=bin_dir/'launchctl'
        launchctl.write_text('#!/bin/sh\nexit 0\n')
        launchctl.chmod(0o755)
        self.env['PATH']=str(bin_dir)+os.pathsep+os.environ['PATH']
        (self.state/'jobs.sqlite3').write_bytes(b'keep registry')
        result=self.run_script('mac_launch_agent.sh','install')
        self.assertEqual(result.returncode,0,result.stderr)
        data=plistlib.loads((self.root/'agent.plist').read_bytes())
        self.assertEqual(data['WorkingDirectory'],str(self.app))
        self.assertEqual(data['ProgramArguments'][0],str(self.app/'.venv/bin/python'))
        self.assertEqual((self.state/'jobs.sqlite3').read_bytes(),b'keep registry')

    def test_stop_and_status_do_not_require_a_new_application_environment(self):
        bin_dir=self.root/'bin'
        bin_dir.mkdir()
        launchctl=bin_dir/'launchctl'
        launchctl.write_text('#!/bin/sh\nexit 0\n')
        launchctl.chmod(0o755)
        self.env['PATH']=str(bin_dir)+os.pathsep+os.environ['PATH']
        for action in ['stop','status']:
            result=self.run_script('mac_launch_agent.sh',action)
            self.assertEqual(result.returncode,0,result.stderr)
        self.assertFalse(self.app.exists())


if __name__=='__main__':
    unittest.main()
