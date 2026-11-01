"""One bounded local HTTP client. Killing it never kills the model service."""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from workbench.io_utils import write_json
from workbench.llm_providers import _NoRedirect, endpoint_url


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    endpoint = endpoint_url(request["endpoint"])
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    req = urllib.request.Request(endpoint + request["path"],
        data=json.dumps(request["body"], ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with opener.open(req, timeout=request["timeout"]) as response:
            raw = response.read(8 * 1024**2 + 1)
        result = {"completed": True, "payload": raw.decode("utf-8") if len(raw) <= 8 * 1024**2 else "",
                  "error_code": "INVALID_RESPONSE" if len(raw) > 8 * 1024**2 else ""}
    except urllib.error.HTTPError as exc:
        result = {"completed": True, "error_code": "MODEL_MISSING" if exc.code == 404 else "PROVIDER_UNAVAILABLE",
                  "http_status": exc.code}
    except (OSError, ValueError, urllib.error.URLError) as exc:
        result = {"completed": isinstance(getattr(exc, "reason", None), ConnectionRefusedError),
                  "error_code": "PROVIDER_UNAVAILABLE", "error_type": type(exc).__name__}
    write_json(Path(request["output"]), result)


if __name__ == "__main__":
    main()
