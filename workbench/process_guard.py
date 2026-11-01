"""Independent guardian for an owned process group; survives caller death."""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from workbench.resources import group_rss, process_alive, process_birth
from workbench.io_utils import write_json


def live_descendants():
    probe = subprocess.Popen(["ps", "-axo", "pid=,pgid=,stat="], stdout=subprocess.PIPE, text=True)
    rows = probe.communicate(timeout=3)[0].splitlines()
    return [int(f[0]) for row in rows if len(f := row.split()) == 3 and int(f[1]) == os.getpgrp()
            and int(f[0]) not in {os.getpid(), probe.pid} and not f[2].startswith("Z")]


def cleanup_group():
    if not live_descendants():
        return
    os.killpg(os.getpgrp(), signal.SIGTERM)
    until = time.monotonic() + 3
    while live_descendants() and time.monotonic() < until:
        time.sleep(.05)
    # This guardian keeps the session/group identity alive while killing its
    # own remaining members, so a reused unrelated PID cannot join this group.
    for pid in live_descendants():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    # Keep the inherited lease until all live members are actually gone.
    # SIGKILL delivery alone is not a cleanup acknowledgement.
    while live_descendants():
        time.sleep(.05)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--owner", type=int, required=True)
    parser.add_argument("--birth", required=True)
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--max-rss", type=int, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    started = time.monotonic()
    child = subprocess.Popen(command, pass_fds=(args.fd,), env={**os.environ, "NEXTECHO_SUPERVISOR_PID": str(os.getpid()),
                             "NEXTECHO_INHERITED_LEASE_FD": str(args.fd)})
    try:
        write_json(args.result.with_suffix(".owner.json"), {"pid": child.pid, "birth": process_birth(child.pid), "pgid": os.getpgrp()})
        while child.poll() is None:
            elapsed = time.monotonic() - started
            owner_alive = process_alive(args.owner, args.birth)
            rss = group_rss(os.getpgrp())
            trigger = ("termination_signal" if stopped else "owner_identity_lost" if not owner_alive else
                       "deadline" if elapsed > args.timeout else "group_rss" if rss > args.max_rss else "")
            if trigger:
                code = "STAGE_TIMEOUT" if trigger == "deadline" else "RESOURCE_LIMIT"
                write_json(args.result, {"error_code": code, "trigger": trigger, "owner_alive": owner_alive,
                    "termination_requested": stopped, "group_rss_bytes": rss, "max_rss_bytes": args.max_rss,
                    "elapsed_seconds": elapsed, "timeout_seconds": args.timeout})
                return 70
            time.sleep(.2)
        return child.returncode
    finally:
        cleanup_group()
        child.wait(timeout=2)


if __name__ == "__main__":
    raise SystemExit(main())
