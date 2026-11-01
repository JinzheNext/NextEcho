#!/usr/bin/env python3
"""Compare synthetic text metrics to existing external libraries, without installs.

Run in an environment which already has RapidFuzz and JiWER. Neither library
is imported by the production scorer. No audio, private text or model is used.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import random
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; preserve previous verification")
    import jiwer
    from rapidfuzz.distance import Levenshtein
    from workbench.transcript_metrics import edit_distance, score_characters

    randomizer = random.Random(910731)
    failures, cases = [], []
    def normalized(text):
        return "".join(c for c in unicodedata.normalize("NFKC", text) if not c.isspace())
    texts = [("", ""), ("", "甲"), ("3.5", "35"), (" Ａ３．５\n繁體", "A35繁体")]
    texts += [("".join(randomizer.choices("甲乙丙丁abc.３５ \n", k=randomizer.randrange(90))),
               "".join(randomizer.choices("甲乙丙丁abc.３５ \n", k=randomizer.randrange(90)))) for _ in range(2000)]
    for index, (ref, hyp) in enumerate(texts):
        a, b = normalized(ref), normalized(hyp)
        expected = Levenshtein.distance(a, b)
        jiwer_result = jiwer.process_characters(a, b)
        jiwer_errors = jiwer_result.substitutions + jiwer_result.insertions + jiwer_result.deletions
        actual = edit_distance(a, b)
        if actual != expected or actual != jiwer_errors:
            failures.append({"kind": "CER", "case": index, "actual": actual, "rapidfuzz": expected, "jiwer": jiwer_errors})
        cases.append({"kind": "CER", "case": index, "reference_chars": len(a), "hypothesis_chars": len(b), "errors": actual})
    for index in range(160):
        refs = ["".join(randomizer.choices("甲乙丙丁", k=randomizer.randrange(25))) for _ in range(randomizer.randrange(6))]
        hyps = ["".join(randomizer.choices("甲乙丙丁", k=randomizer.randrange(25))) for _ in range(randomizer.randrange(6))]
        size = max(len(refs), len(hyps))
        padded_r = refs + [""] * (size - len(refs))
        padded_h = hyps + [""] * (size - len(hyps))
        expected = min((sum(Levenshtein.distance(padded_r[r], padded_h[h]) for r, h in enumerate(order))
                        for order in itertools.permutations(range(size))), default=0)
        r = [{"speaker_id": f"r{i}", "text": text} for i, text in enumerate(refs)]
        h = [{"speaker_id": f"h{i}" if i else None, "text": text} for i, text in enumerate(hyps)]
        actual = score_characters(r, h)["cpcer"]["errors"]
        if actual != expected:
            failures.append({"kind": "cpCER", "case": index, "actual": actual, "exhaustive_rapidfuzz": expected})
        cases.append({"kind": "cpCER", "case": index, "reference_streams": len(refs), "hypothesis_streams": len(hyps), "errors": actual})
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        Path(__file__).resolve(), ROOT / "workbench/transcript_metrics.py", ROOT / "workbench/speaker_benchmark.py")}
    report = {
        "schema_version": 1, "synthetic_only": True, "model_release_approved": False,
        "seed": 910731, "character_cases": len(texts), "cpcer_cases": 160,
        "reference_versions": {n: importlib.metadata.version(n) for n in ("rapidfuzz", "jiwer")},
        "reference_methods": ["RapidFuzz exact Levenshtein", "JiWER character I/D/S total", "exhaustive speaker permutations with RapidFuzz costs"],
        "normalization": "independent NFKC and Unicode whitespace removal; punctuation/script/case retained",
        "tested_sources": hashes, "mismatches": failures, "cases": cases,
        "scope": "unit-cost character edits and recording-wide cpCER assignment only; not MeetEval execution, timing accuracy, human reference correctness or model quality",
        "new_model_inference_requests": 0, "Pro_messages_sent": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "character_cases": len(texts), "cpcer_cases": 160, "mismatches": len(failures)}))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
