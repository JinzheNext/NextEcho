from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workbench.speaker_benchmark import evaluate_manifest


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: evaluate_speakers.py manifest.json", file=sys.stderr)
        return 2
    try:
        result = evaluate_manifest(Path(argv[1]))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"speaker benchmark failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
