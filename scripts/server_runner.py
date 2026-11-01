from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import RUNS_ROOT, app


def main() -> None:
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    host = os.environ.get("NEXTECHO_HOST", "127.0.0.1")
    port = int(os.environ.get("NEXTECHO_PORT", "8765"))
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
