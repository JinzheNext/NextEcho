"""Relay media output with an exact byte cap inside the owned guardian group."""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from workbench.io_utils import write_json


def execute(request):
    maximum = request["max_output_bytes"]
    total = 0
    process = None
    fd = os.environ.get("NEXTECHO_INHERITED_LEASE_FD", "")
    inherited = (int(fd),) if os.name == "posix" and fd.isdigit() else ()
    with (
        open(request["stdout_path"], "wb") as stdout,
        open(request["stderr_path"], "wb") as stderr,
    ):
        stdin = open(request["stdin_path"], "rb") if request.get("stdin_path") else None
        try:
            process = subprocess.Popen(
                request["command"],
                cwd=request["cwd"],
                stdin=stdin if stdin else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **({"pass_fds": inherited} if os.name == "posix" else {}),
            )
        except OSError as exc:
            return {
                "error_type": type(exc).__name__,
                "errno": exc.errno,
                "output_bytes": 0,
            }
        finally:
            if stdin:
                stdin.close()
        try:
            with selectors.DefaultSelector() as selector:
                for stream, sink in (
                    (process.stdout, stdout),
                    (process.stderr, stderr),
                ):
                    selector.register(stream, selectors.EVENT_READ, sink)
                while selector.get_map():
                    for key, _ in selector.select(0.1):
                        data = os.read(key.fd, 65536)
                        if not data:
                            selector.unregister(key.fileobj)
                            continue
                        remaining = maximum - total
                        key.data.write(data[:remaining])
                        total += len(data[:remaining])
                        if len(data) > remaining:
                            # Exit the relay; the independent guardian then
                            # reaps this command and all its owned descendants.
                            return {"error_code": "OUTPUT_LIMIT", "output_bytes": total}
                while process.poll() is None:
                    time.sleep(0.05)
                return {"returncode": process.returncode, "output_bytes": total}
        finally:
            process.stdout.close()
            process.stderr.close()
            # Do not wait on an uncooperative child here: guardian owns cleanup.
            if process.poll() is not None:
                process.wait()


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    write_json(Path(request["response_path"]), execute(request))


if __name__ == "__main__":
    main()
