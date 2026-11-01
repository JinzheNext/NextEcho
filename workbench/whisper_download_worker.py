"""Standalone bounded HTTPS transport. No account credentials or model inference."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


class HTTPSOnlyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlparse(newurl).scheme != "https":
            raise ValueError("non-HTTPS model redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(spec: dict, destination: Path, *, opener=None) -> dict:
    if urlparse(spec["url"]).scheme != "https":
        raise ValueError("non-HTTPS model URL")
    expected = int(spec["bytes"])
    if not 64 <= expected <= 16 * 1024**3:
        raise ValueError("invalid model size budget")
    opener = opener or urllib.request.build_opener(HTTPSOnlyRedirect())
    digest, count = hashlib.sha256(), 0
    request = urllib.request.Request(spec["url"], headers={"User-Agent": "NextEcho-model-prepare/1"})
    with opener.open(request, timeout=30) as response, destination.open("wb") as output:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) != expected:
            raise ValueError("model Content-Length mismatch")
        while data := response.read(min(1024**2, expected - count + 1)):
            count += len(data)
            if count > expected:
                raise ValueError("model exceeds size budget")
            output.write(data)
            digest.update(data)
        if count != expected or digest.hexdigest() != spec["sha256"]:
            raise ValueError("model bytes/digest mismatch")
        output.flush()
        os.fsync(output.fileno())
    return {"bytes": count, "sha256": digest.hexdigest()}


def main():
    try:
        request = json.loads(Path(sys.argv[1]).read_text())
        print(json.dumps(download(request["spec"], Path(request["destination"]))))
    except Exception as exc:
        # Error type is useful; URLs/redirect signatures or local request data are not.
        print("model transport failed: " + type(exc).__name__, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
