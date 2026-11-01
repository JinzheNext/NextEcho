from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def normalize(text: str) -> str:
    return re.sub(r"[^\w\u3400-\u9fff]+", "", text.lower())


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current = [i]
        for j, hyp_char in enumerate(hypothesis, start=1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ref_char != hyp_char)))
        previous = current
    return previous[-1]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: evaluate_asr.py manifest.json", file=sys.stderr)
        return 2
    manifest = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    totals: dict[str, dict[str, int]] = {}
    for item in manifest:
        reference = normalize(Path(item["reference"]).read_text(encoding="utf-8"))
        for engine, hypothesis_path in (item.get("hypotheses") or {}).items():
            hypothesis = normalize(Path(hypothesis_path).read_text(encoding="utf-8"))
            row = totals.setdefault(engine, {"distance": 0, "reference_chars": 0})
            row["distance"] += edit_distance(reference, hypothesis)
            row["reference_chars"] += len(reference)
    result = {
        engine: {**row, "cer": round(row["distance"] / max(row["reference_chars"], 1), 6)}
        for engine, row in totals.items()
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
