"""Optional independent scorer parity; requires a local dscore checkout + NumPy/SciPy.

No model, audio, downloads or reference code are bundled by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workbench.diarization_metrics import evaluate_continuous_regions  # noqa: E402
from workbench.speaker_benchmark import SpeakerRegion as R  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dscore-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.dscore_root.expanduser().resolve()
    sys.path.insert(0, str(root))
    import numpy as np
    from scorelib import metrics as official
    from scorelib.rttm import write_rttm
    from scorelib.turn import Turn
    from scorelib.uem import UEM, write_uem

    cases = [
        ("outside-reference", [R(1000, 2000, "A")], [R(0, 3000, "X")], [[0, 3000]]),
        ("non-grid-tail", [R(3, 27, "A")], [R(4, 28, "X")], [[0, 31]]),
        ("overlap", [R(0, 1000, "A"), R(500, 1500, "B")], [R(0, 1500, "X")], [[0, 1500]]),
        ("jer-distinct-mapping", [R(0, 1000, "A"), R(0, 100, "B")], [R(0, 101, "X")], [[0, 1000]]),
        ("cross-window-swap", [R(0, 10, "A"), R(10, 20, "B"), R(30, 40, "A"), R(40, 50, "B")], [R(0, 10, "X"), R(10, 20, "Y"), R(30, 40, "Y"), R(40, 50, "X")], [[0, 20], [30, 50]]),
    ]
    rng = random.Random(20261019)
    for case in range(40):
        reference = [R(start, rng.randrange(start + 800, 5001), label) for start, label in ((rng.randrange(1200), "A"), (rng.randrange(1200), "B"))]
        hypothesis = [R(start, rng.randrange(start + 800, 5001), label) for start, label in ((rng.randrange(1200), "X"), (rng.randrange(1200), "Y"), (rng.randrange(1200), "Z"))]
        cases.append((f"random-{case:02d}", reference, hypothesis, [[0, 5000]]))

    results = []
    failures = []
    with tempfile.TemporaryDirectory(prefix="nextecho-scorer-parity-") as directory:
        work = Path(directory)
        for name, reference, hypothesis, uem in cases:
            for collar, skip in ((0, False), (100, False), (100, True)) if name.startswith("random") or name == "overlap" else ((0, False),):
                actual = evaluate_continuous_regions(reference, hypothesis, uem_ms=uem, collar_ms=collar, skip_overlap=skip)
                def turns(regions):
                    return [Turn(r.start_ms / 1000, offset=r.end_ms / 1000, speaker_id=r.speaker, file_id="probe") for r in regions]
                write_rttm(str(work / "ref.rttm"), turns(reference))
                write_rttm(str(work / "hyp.rttm"), turns(hypothesis))
                write_uem(str(work / "all.uem"), UEM({"probe": [(s / 1000, e / 1000) for s, e in uem]}))
                # md-eval -c is EACH-side width; our collar_ms is TOTAL width.
                command = ["perl", str(root / "scorelib/md-eval-22.pl"), "-af", "-r", str(work / "ref.rttm"), "-s", str(work / "hyp.rttm"), "-u", str(work / "all.uem"), "-c", str(collar / 2000)]
                if skip:
                    command.append("-1")
                process = subprocess.run(command, capture_output=True, text=True, timeout=30)
                observed = {}
                for output_key, label in (("reference_speaker_ms", "SCORED SPEAKER"), ("miss_ms", "MISSED SPEAKER"), ("false_alarm_ms", "FALARM SPEAKER"), ("confusion_ms", "SPEAKER ERROR")):
                    values = re.findall(re.escape(label) + r" TIME =\s*([\d.]+)", process.stdout)
                    if not values:
                        raise RuntimeError(f"md-eval missing {label}: {process.stderr[-1000:]}")
                    observed[output_key] = float(values[-1]) * 1000
                # Independently rasterize at 0.5 ms for these integer/half-ms
                # synthetic boundaries, then call dscore's ORIGINAL JER code.
                times = np.arange(max(e for _, e in uem) * 2) / 2 + 0.25
                refs, hyps = sorted({r.speaker for r in reference}), sorted({r.speaker for r in hypothesis})
                ref_matrix = np.array([[any(r.speaker == label and r.start_ms <= t < r.end_ms for r in reference) for label in refs] for t in times], dtype=float)
                hyp_matrix = np.array([[any(r.speaker == label and r.start_ms <= t < r.end_ms for r in hypothesis) for label in hyps] for t in times], dtype=float)
                keep = np.array([any(s <= t < e for s, e in uem) for t in times])
                if collar:
                    keep &= np.array([all(abs(t - b) >= collar / 2 for r in reference for b in (r.start_ms, r.end_ms)) for t in times])
                if skip:
                    keep &= ref_matrix.sum(axis=1) <= 1
                ref_matrix, hyp_matrix = ref_matrix[keep], hyp_matrix[keep]
                ref_matrix = ref_matrix[:, ref_matrix.sum(axis=0) > 0]
                hyp_matrix = hyp_matrix[:, hyp_matrix.sum(axis=0) > 0]
                _, jer = official.jer({"probe": ref_matrix.sum(axis=0) / 2}, {"probe": hyp_matrix.sum(axis=0) / 2}, {"probe": ref_matrix.T @ hyp_matrix / 2})
                mismatch = {key: [actual[key], value] for key, value in observed.items() if abs(actual[key] - value) > 1e-5}
                if abs(actual["jer"] - jer / 100) > 1e-10:
                    mismatch["jer"] = [actual["jer"], jer / 100]
                record = {"case": name, "reference": [r.__dict__ for r in reference], "hypothesis": [r.__dict__ for r in hypothesis], "uem_ms": uem, "collar_ms": collar, "skip_overlap": skip, "actual": actual, "md_eval_components_ms": observed, "dscore_jer": jer / 100, "mismatch": mismatch}
                results.append(record)
                if mismatch:
                    failures.append(record)
    report = {
        "metric_protocol": "continuous-ms-v2", "synthetic_only": True,
        "reference_repository": "https://github.com/nryant/dscore", "reference_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "reference_files_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ("scorelib/metrics.py", "scorelib/md-eval-22.pl")},
        "implementation_sha256": {name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() for name in ("workbench/diarization_metrics.py", "workbench/speaker_benchmark.py", "scripts/verify_speaker_metrics.py")},
        "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "intervaltree")},
        "jer_scope": "Original dscore IoU assignment/aggregation with independently generated exact duration matrices, not its default 10ms rasterization.",
        "der_scope": "Original md-eval-22 continuous components; -c converted from total width to each-side width. Same explicit UEM/overlap setting.",
        "cases": len(results), "failures": len(failures), "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"cases": len(results), "failures": len(failures), "report": str(args.output.resolve())}))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
