"""Real small processes/media; no inference, credentials, or external network."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import wave
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from workbench.contracts import PipelineError
from workbench.media_commands import run_media_command
from workbench.resources import ExecutionLease, GIB, group_running, model_policy


def good_sample():
    return {"available_bytes": 100 * GIB, "swap_used_bytes": 0, "pressure": "normal"}


class ResourceFloorTests(unittest.TestCase):
    def test_unknown_or_nonfinite_samples_are_not_reported_as_free_or_normal(self):
        from types import SimpleNamespace
        from workbench.asr import memory_snapshot, _read_memory_pressure
        with patch("workbench.asr.subprocess.run", return_value=SimpleNamespace(stdout="unrecognized", stderr="")):
            self.assertEqual(_read_memory_pressure()[0], "unknown")
        with patch("workbench.asr.platform.system", return_value="Darwin"), patch(
            "workbench.asr.subprocess.run", side_effect=subprocess.TimeoutExpired("sampler", 3)
        ) as run:
            sample = memory_snapshot().to_dict()
        self.assertIsNone(sample["available_bytes"])
        self.assertIsNone(sample["swap_used_bytes"])
        self.assertEqual(sample["pressure"], "unknown")
        self.assertTrue(all(call.kwargs.get("timeout") == 3 for call in run.call_args_list))
        policy = model_policy(1, GIB, 5, {"hardware": {}})
        for field in ("available_bytes", "swap_used_bytes"):
            for bad in (None, float("nan"), float("inf"), -1, "100"):
                invalid = {**good_sample(), field: bad}
                with self.assertRaises(PipelineError):
                    policy.preflight(invalid)
                self.assertTrue(policy.violated({**invalid, "worker_rss_bytes": 1}, good_sample()))

    def test_production_floor_and_nonfinite_budgets(self):
        for reserve, expected in [(0, 3), (0.5, 3), (3, 3), (4, 4)]:
            policy = model_policy(
                1, GIB, 10, {"hardware": {"reserve_memory_gb": reserve}}
            )
            self.assertEqual(policy.reserve_bytes, expected * GIB)
            with self.assertRaises(PipelineError):
                policy.preflight(
                    {
                        "available_bytes": expected * GIB,
                        "pressure": "normal",
                        "swap_used_bytes": 0,
                    }
                )
        for bad in [float("nan"), float("inf"), -1, "invalid"]:
            for key in ["reserve_memory_gb", "max_swap_delta_gb"]:
                with self.assertRaises(PipelineError) as error:
                    model_policy(1, GIB, 10, {"hardware": {key: bad}})
                self.assertEqual(error.exception.code, "INVALID_INPUT")


@unittest.skipUnless(os.name == "posix", "POSIX owned media")
class MediaCommandTests(unittest.TestCase):
    def context(self, root):
        return patch.dict(
            os.environ, {"NEXTECHO_RUNTIME_DIR": str(root), "NEXTECHO_JOB_ID": ""}
        )

    def test_bounded_stdout_stdin_errors_and_private_diagnostics(self):
        with (
            tempfile.TemporaryDirectory() as d,
            self.context(Path(d)),
            patch("workbench.media_commands.sample_resources", side_effect=good_sample),
        ):
            result = run_media_command(
                [
                    sys.executable,
                    "-c",
                    'import sys; print(sys.stdin.read());print("notice",file=sys.stderr)',
                ],
                input="原文",
                timeout=5,
            )
            self.assertEqual(result.stdout, "原文\n")
            self.assertEqual(result.stderr, "notice\n")
            with self.assertRaises(subprocess.CalledProcessError) as error:
                run_media_command(
                    [
                        sys.executable,
                        "-c",
                        'import sys;print("cookie: SECRET",file=sys.stderr);sys.exit(23)',
                    ],
                    timeout=5,
                )
            self.assertEqual(error.exception.returncode, 23)
            self.assertNotIn("SECRET", error.exception.stderr)
            with self.assertRaises(FileNotFoundError):
                run_media_command(["/nonexistent-nextecho-media-tool"], timeout=5)
            for log in (Path(d) / "media-logs").glob("*.diagnostic.json"):
                self.assertGreaterEqual(
                    json.loads(log.read_text())["effective_reserve_bytes"], 3 * GIB
                )
            self.assertEqual(list((Path(d) / "requests").iterdir()), [])

    def test_output_cap_kills_the_owned_group(self):
        with (
            tempfile.TemporaryDirectory() as d,
            self.context(Path(d)),
            patch("workbench.media_commands.sample_resources", side_effect=good_sample),
        ):
            root = Path(d)
            pgid = root / "pgid"
            code = 'import os,sys;from pathlib import Path;Path(sys.argv[1]).write_text(str(os.getpgrp()));\nwhile True: os.write(1,b"x"*65536)'
            with self.assertRaises(PipelineError) as error:
                run_media_command(
                    [sys.executable, "-c", code, str(pgid)],
                    timeout=5,
                    max_output_bytes=1024,
                )
            self.assertEqual(error.exception.code, "OUTPUT_LIMIT")
            self.assertFalse(group_running(int(pgid.read_text())))
            diagnostic = json.loads(
                next((root / "media-logs").glob("*.diagnostic.json")).read_text()
            )
            self.assertEqual(diagnostic["output_bytes"], 1024)
            with ExecutionLease(root, timeout=0.2):
                pass

    def test_cancellation_does_not_release_lease_during_descendant_cleanup(self):
        with (
            tempfile.TemporaryDirectory() as d,
            self.context(Path(d)),
            patch("workbench.media_commands.sample_resources", side_effect=good_sample),
        ):
            root = Path(d)
            ready = root / "ready"
            terminated = root / "terminated"
            pgid = root / "pgid"
            child_code = 'import signal,sys,time;from pathlib import Path;signal.signal(signal.SIGTERM,lambda *a:Path(sys.argv[1]).write_text("term"));print("ready",flush=True);time.sleep(20)'
            command_code = 'import os,subprocess,sys,time;from pathlib import Path;p=subprocess.Popen([sys.executable,"-c",sys.argv[1],sys.argv[2]],stdout=subprocess.PIPE,text=True);p.stdout.readline();Path(sys.argv[3]).write_text(str(os.getpgrp()));Path(sys.argv[4]).write_text("ready");time.sleep(20)'
            observations = []

            def observe_lock():
                until = time.monotonic() + 10
                while not terminated.exists() and time.monotonic() < until:
                    time.sleep(0.02)
                if not terminated.exists():
                    observations.append("no-term")
                    return
                try:
                    with ExecutionLease(root, timeout=0.15):
                        observations.append("released-early")
                except TimeoutError:
                    observations.append("held-until-reaped")

            unrelated = subprocess.Popen(
                [sys.executable, "-c", "import time;time.sleep(20)"]
            )
            watcher = threading.Thread(target=observe_lock)
            watcher.start()
            try:
                with patch(
                    "workbench.media_commands.current_cancelled",
                    side_effect=ready.exists,
                ):
                    with self.assertRaises(PipelineError) as error:
                        run_media_command(
                            [
                                sys.executable,
                                "-c",
                                command_code,
                                child_code,
                                str(terminated),
                                str(pgid),
                                str(ready),
                            ],
                            timeout=10,
                        )
                self.assertEqual(error.exception.code, "CANCELLED")
                watcher.join(timeout=5)
                self.assertEqual(observations, ["held-until-reaped"])
                self.assertFalse(group_running(int(pgid.read_text())))
                self.assertIsNone(unrelated.poll())
                with ExecutionLease(root, timeout=0.3):
                    pass
            finally:
                unrelated.terminate()
                unrelated.wait(timeout=3)
                watcher.join(timeout=5)

    def test_media_timeout_cleans_up(self):
        with (
            tempfile.TemporaryDirectory() as d,
            self.context(Path(d)),
            patch("workbench.media_commands.sample_resources", side_effect=good_sample),
        ):
            root = Path(d)
            pgid = root / "pgid"
            code = "import os,sys,time;from pathlib import Path;Path(sys.argv[1]).write_text(str(os.getpgrp()));time.sleep(20)"
            with self.assertRaises(PipelineError) as error:
                run_media_command([sys.executable, "-c", code, str(pgid)], timeout=0.6)
            self.assertEqual(error.exception.code, "STAGE_TIMEOUT")
            self.assertFalse(group_running(int(pgid.read_text())))
            with ExecutionLease(root, timeout=0.3):
                pass

    def test_queued_media_cancel_does_not_spawn_a_command(self):
        with tempfile.TemporaryDirectory() as d, self.context(Path(d)), patch(
            "workbench.media_commands.sample_resources", side_effect=good_sample
        ):
            root = Path(d)
            cancelled = threading.Event()
            result = []

            def wait_for_lease():
                try:
                    run_media_command([sys.executable, "-c", "pass"], timeout=5)
                except PipelineError as exc:
                    result.append(exc.code)

            with patch("workbench.media_commands.current_cancelled", side_effect=cancelled.is_set):
                with ExecutionLease(root):
                    waiter = threading.Thread(target=wait_for_lease)
                    waiter.start()
                    time.sleep(.15)
                    cancelled.set()
                    waiter.join(timeout=2)
                    self.assertFalse(waiter.is_alive())
            self.assertEqual(result, ["CANCELLED"])
            self.assertEqual(list((root / "media-logs").glob("*.guard.owner.json")), [])

    def test_media_caller_relay_and_guardian_death_reap_the_actual_command(self):
        caller_code = '''import json,sys
from pathlib import Path
import workbench.media_commands as media
root=Path(sys.argv[1])
media.sample_resources=lambda:{'available_bytes':10**12,'pressure':'normal','swap_used_bytes':0}
command=[sys.executable,'-c',"import os,sys,time,json;from pathlib import Path;Path(sys.argv[1]).write_text(json.dumps({'pgid':os.getpgrp(),'pid':os.getpid()}));time.sleep(20)",str(root/'media.json')]
try:media.run_media_command(command,timeout=15,config={'hardware':{}})
except Exception as exc:(root/'error.json').write_text(json.dumps({'code':getattr(exc,'code',type(exc).__name__)}))
'''
        for target in ("caller", "relay", "guardian"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                caller = subprocess.Popen([sys.executable, "-c", caller_code, d],
                    env={**os.environ, "NEXTECHO_RUNTIME_DIR": d, "NEXTECHO_JOB_ID": ""})
                pgid = None
                try:
                    deadline = time.monotonic() + 8
                    while not (root / "media.json").exists() and time.monotonic() < deadline:
                        time.sleep(.03)
                    self.assertTrue((root / "media.json").exists())
                    pgid = json.loads((root / "media.json").read_text())["pgid"]
                    owner = json.loads(next((root / "media-logs").glob("*.guard.owner.json")).read_text())
                    pid = caller.pid if target == "caller" else owner["pid"] if target == "relay" else pgid
                    os.kill(pid, signal.SIGKILL)
                    caller.wait(timeout=10)
                    deadline = time.monotonic() + 6
                    while group_running(pgid) and time.monotonic() < deadline:
                        time.sleep(.05)
                    self.assertFalse(group_running(pgid), target)
                    with ExecutionLease(root, timeout=.5):
                        pass
                    if target != "caller":
                        self.assertIn(json.loads((root / "error.json").read_text())["code"], {"WORKER_FAILED", "RESOURCE_LIMIT"})
                finally:
                    if caller.poll() is None:
                        caller.kill()
                        caller.wait(timeout=3)
                    if pgid is not None and group_running(pgid):
                        os.killpg(pgid, signal.SIGKILL)

    @unittest.skipUnless(
        shutil.which("ffmpeg") and shutil.which("ffprobe") and shutil.which("curl"),
        "local media tools",
    )
    def test_real_loopback_download_extract_segment_and_probe(self):
        from workbench.transcription import (
            _download_direct_media,
            extract_audio,
            extract_audio_segment,
            probe_audio_duration_ms,
        )

        with (
            tempfile.TemporaryDirectory() as d,
            self.context(Path(d) / "runtime"),
            patch("workbench.media_commands.sample_resources", side_effect=good_sample),
        ):
            root = Path(d)
            source = root / "source.wav"
            download = root / "downloads"
            download.mkdir()
            with wave.open(str(source), "wb") as handle:
                handle.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                handle.writeframes(b"\0\0" * 16000)

            class QuietHandler(SimpleHTTPRequestHandler):
                def log_message(self, *args):
                    pass

            server = ThreadingHTTPServer(
                ("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(root))
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                fetched = _download_direct_media(
                    f"http://127.0.0.1:{server.server_port}/source.wav", download
                )
                self.assertEqual(
                    hashlib.sha256(source.read_bytes()).digest(),
                    hashlib.sha256(fetched.read_bytes()).digest(),
                )
                extracted = extract_audio(
                    source_path=fetched,
                    target_path=root / "extracted.wav",
                    ffmpeg_bin=shutil.which("ffmpeg"),
                    max_seconds=0,
                )
                self.assertEqual(probe_audio_duration_ms(extracted), 1000)
                segment = extract_audio_segment(
                    source_path=extracted,
                    target_path=root / "segment.wav",
                    offset_ms=200,
                    duration_ms=400,
                )
                self.assertEqual(probe_audio_duration_ms(segment), 400)
                self.assertGreaterEqual(
                    len(list((root / "runtime/media-logs").glob("*.diagnostic.json"))),
                    5,
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
