import json
import os
import shutil
import socket
import sys
import time
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from workbench.config import NextEchoConfig
from workbench.contracts import PipelineError
from workbench.llm_providers import request_json
from workbench.owned_provider import owned_provider
from workbench.resources import group_running


@unittest.skipUnless(os.name == "posix" and shutil.which("lsof"), "POSIX + lsof")
class OwnedProviderTests(unittest.TestCase):
    def test_owned_service_stops_on_exception_and_preserves_external_listener(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            script = root / "fake-service.py"
            script.write_text('''import json,sys
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
class H(BaseHTTPRequestHandler):
 def log_message(self,*a):pass
 def do_GET(self):
  self.send_response(200);self.end_headers();self.wfile.write(json.dumps({'data':[{'id':'fake-model'}]}).encode())
ThreadingHTTPServer(('127.0.0.1',int(sys.argv[1])),H).serve_forever()
''')
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            settings = {"ownership": "owned", "start_command": [sys.executable, str(script), str(port)],
                        "estimated_bytes": 1, "max_rss_bytes": 512 * 1024**2, "timeout_seconds": 10, "startup_timeout_seconds": 5, "sample_seconds": .1}
            config = NextEchoConfig({"hardware": {"reserve_memory_gb": 0}})
            with patch.dict(os.environ, {"NEXTECHO_RUNTIME_DIR": str(root / "runtime")}):
                with self.assertRaisesRegex(ValueError, "fake failure"):
                    with owned_provider(settings, endpoint=f"http://127.0.0.1:{port}", model="fake-model", config=config) as info:
                        self.assertEqual(request_json(f"http://127.0.0.1:{port}", "/v1/models")["data"][0]["id"], "fake-model")
                        pgid = info["owned_pgid"]
                        raise ValueError("fake failure")
                self.assertFalse(group_running(pgid))
                evidence = next((root / "runtime/owned-services").glob("*/resources.json"))
                self.assertTrue(json.loads(evidence.read_text())["cleanup_complete"])
                body_active = False
                def sample():
                    return {"available_bytes": 10**12, "pressure": "critical" if body_active else "normal", "swap_used_bytes": 0}
                with patch("workbench.owned_provider.sample_resources", side_effect=sample):
                    with self.assertRaises(PipelineError) as caught:
                        with owned_provider(settings, endpoint=f"http://127.0.0.1:{port}", model="fake-model", config=config) as info:
                            body_active = True
                            until = time.monotonic() + 3
                            while group_running(info["owned_pgid"]) and time.monotonic() < until:
                                time.sleep(.05)
                            raise PipelineError("MODEL_MISSING", "secondary error after protection stopped service")
                self.assertEqual(caught.exception.code, "RESOURCE_LIMIT")
                body_active = False
                class Handler(BaseHTTPRequestHandler):
                    def log_message(self, *args): pass
                    def do_GET(self):
                        self.send_response(200)
                        self.end_headers()
                        self.wfile.write(b'{"alive":true}')
                server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    with self.assertRaises(PipelineError) as caught:
                        with owned_provider(settings, endpoint=f"http://127.0.0.1:{port}", model="fake-model", config=config):
                            self.fail("adopted external listener")
                    self.assertEqual(caught.exception.code, "OWNERSHIP_CONFLICT")
                    self.assertTrue(request_json(f"http://127.0.0.1:{port}", "/")["alive"])
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)
