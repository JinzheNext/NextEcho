"""Compare RTTM/UEM parsing and 10ms JER to the pinned original dscore.

Run using an existing isolated NumPy/SciPy/dscore environment. This script
downloads nothing, installs nothing and uses only generated synthetic turns.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workbench.diarization_metrics import evaluate_continuous_regions  # noqa: E402
from workbench.speaker_scoring_io import (  # noqa: E402
    _canonical_regions,
    dscore_10ms_jer,
    read_rttm,
    read_uem,
)

COMMIT = "e02f949ac6592279300a2c33d03daf9e0c12fd27"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dscore-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference_root = args.dscore_root.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(reference_root), "rev-parse", "HEAD"], text=True, timeout=10
    ).strip()
    if commit != COMMIT:
        raise ValueError("independent scorer must match the frozen commit")
    sys.path.insert(0, str(reference_root))
    import numpy as np
    from scorelib import metrics
    from scorelib.rttm import load_rttm
    from scorelib.score import turns_to_frames
    from scorelib.turn import merge_turns, trim_turns
    from scorelib.uem import load_uem

    # Decimal seconds are parsed independently from the very same RTTM/UEM
    # bytes by both implementations, including CLI trim/merge preprocessing.
    cases = [
        ("non-grid", [(0.003, 0.024, "A")], [(0.004, 0.024, "X")], [(0, 0.031)]),
        ("zero-sampled", [(0.001, 0.002, "A")], [(0.001, 0.002, "X")], [(0, 0.010)]),
        ("silence", [], [], [(0, 1)]),
        ("no-ref", [], [(0, 1, "X")], [(0, 1)]),
        ("no-hyp", [(0, 1, "A")], [], [(0, 1)]),
        ("outside-uem", [(0, 1, "A"), (2, 1, "B")], [(0, 1, "X")], [(0, 1.001)]),
        (
            "float-grid",
            [(0.29, 0.28, "A"), (0.57, 0.7, "B")],
            [(0.29, 0.28, "X"), (0.57, 0.7, "Y")],
            [(0, 1.27)],
        ),
        (
            "merge-before-collar",
            [(0, 0.4, "A"), (0.2, 0.4, "A"), (0.6, 0.2, "A")],
            [(0, 0.8, "X")],
            [(0, 1)],
        ),
        (
            "uem-cut-collar",
            [(0, 2, "A")],
            [(0.3, 1.4, "X")],
            [(0.25, 0.85), (1.15, 1.75)],
        ),
        (
            "sparse-identity-swap",
            [(0, 0.1, "A"), (0.1, 0.1, "B"), (0.3, 0.1, "A"), (0.4, 0.1, "B")],
            [(0, 0.1, "X"), (0.1, 0.1, "Y"), (0.3, 0.1, "Y"), (0.4, 0.1, "X")],
            [(0, 0.2), (0.3, 0.5)],
        ),
        (
            "nine-clusters",
            [(i * 0.1, 0.1, f"R{i}") for i in range(9)],
            [(i * 0.1, 0.1, f"H{i}") for i in range(9)],
            [(0, 1)],
        ),
    ]
    rng = random.Random(20261020)
    for index in range(80):

        def turns(prefix):
            result = []
            for _ in range(rng.randrange(1, 10)):
                start = rng.randrange(0, 2000000) / 1000000
                duration = rng.randrange(1, 1000000) / 1000000
                result.append((start, duration, prefix + str(rng.randrange(4))))
            return result

        cases.append(
            (
                f"random-{index:02}",
                turns("R"),
                turns("H"),
                [(0, 0.750123), (1.010001, 3.000001)],
            )
        )
    results, failures = [], []
    with tempfile.TemporaryDirectory(prefix="nextecho-10ms-parity-") as temp:
        work = Path(temp)
        for name, reference, hypothesis, uem in cases:
            for label, rows in (("ref", reference), ("hyp", hypothesis)):
                (work / (label + ".rttm")).write_text(
                    "".join(
                        f"SPEAKER probe 1 {start:.6f} {dur:.6f} <NA> <NA> {speaker} <NA> <NA>\n"
                        for start, dur, speaker in rows
                    )
                )
            (work / "all.uem").write_text(
                "".join(f"probe 1 {a:.6f} {b:.6f}\n" for a, b in uem)
            )
            ref, _ = read_rttm(work / "ref.rttm")
            hyp, _ = read_rttm(work / "hyp.rttm")
            uems, _ = read_uem(work / "all.uem")
            ref, hyp, windows = (
                ref.get(("probe", "1"), []),
                hyp.get(("probe", "1"), []),
                uems["probe", "1"],
            )
            actual = dscore_10ms_jer(ref, hyp, uem_ms=windows)
            official_uem = load_uem(str(work / "all.uem"))
            warning = io.StringIO()
            with contextlib.redirect_stderr(warning):
                official_ref = merge_turns(
                    trim_turns(load_rttm(str(work / "ref.rttm"))[0], official_uem)
                )
                official_hyp = merge_turns(
                    trim_turns(load_rttm(str(work / "hyp.rttm"))[0], official_uem)
                )
            rframes = turns_to_frames(official_ref, official_uem["probe"])
            hframes = turns_to_frames(official_hyp, official_uem["probe"])
            reference_jer, original_error = None, None
            with np.errstate(invalid="ignore"):
                try:
                    _, reference_jer = metrics.jer(
                        {"probe": rframes.sum(axis=0)},
                        {"probe": hframes.sum(axis=0)},
                        {"probe": metrics.contingency_matrix(rframes, hframes)},
                    )
                    reference_jer /= 100
                except ValueError as exc:
                    if "invalid numeric" not in str(exc):
                        raise
                    original_error = (
                        "assignment contains invalid numeric entries (zero union)"
                    )
            mismatch = []
            if original_error:
                if actual["defined"]:
                    mismatch.append(
                        "official zero-union is undefined, local implementation reported a score"
                    )
            elif not actual["defined"] or abs(reference_jer - actual["jer"]) > 1e-12:
                mismatch.append("JER differs")
            if sorted(rframes.sum(axis=0).tolist()) != sorted(
                actual["reference_speaker_frames"].values()
            ):
                mismatch.append("reference sampled durations differ")
            if sorted(hframes.sum(axis=0).tolist()) != sorted(
                actual["hypothesis_speaker_frames"].values()
            ):
                mismatch.append("hypothesis sampled durations differ")
            der_rows = []
            for collar, skip in ((0, False), (100, False), (100, True)):
                try:
                    continuous = evaluate_continuous_regions(
                        _canonical_regions(ref, windows),
                        _canonical_regions(hyp, windows),
                        uem_ms=windows,
                        collar_ms=collar,
                        skip_overlap=skip,
                        allow_fractional_ms=True,
                    )
                except ValueError as exc:
                    if "no scored time" in str(exc):
                        der_rows.append(
                            {
                                "collar_ms": collar,
                                "skip_overlap": skip,
                                "scored_time": "none",
                                "compared": False,
                            }
                        )
                        continue
                    raise
                command = [
                    "perl",
                    str(reference_root / "scorelib/md-eval-22.pl"),
                    "-af",
                    "-r",
                    str(work / "ref.rttm"),
                    "-s",
                    str(work / "hyp.rttm"),
                    "-u",
                    str(work / "all.uem"),
                    "-c",
                    str(collar / 2000),
                ]
                # Use dscore CLI-preprocessed turns for the DER subprocess too.
                from scorelib.rttm import write_rttm

                write_rttm(str(work / "ref-trim.rttm"), official_ref, n_digits=9)
                write_rttm(str(work / "hyp-trim.rttm"), official_hyp, n_digits=9)
                command[command.index("-r") + 1] = str(work / "ref-trim.rttm")
                command[command.index("-s") + 1] = str(work / "hyp-trim.rttm")
                if skip:
                    command.append("-1")
                process = subprocess.run(
                    command, capture_output=True, text=True, timeout=15
                )
                observed = {}
                for key, text in (
                    ("reference_speaker_ms", "SCORED SPEAKER"),
                    ("miss_ms", "MISSED SPEAKER"),
                    ("false_alarm_ms", "FALARM SPEAKER"),
                    ("confusion_ms", "SPEAKER ERROR"),
                ):
                    values = re.findall(
                        re.escape(text) + r" TIME =\s*([\d.]+)", process.stdout
                    )
                    if not values:
                        if not official_ref or not official_hyp:
                            observed = None
                            der_rows.append(
                                {
                                    "collar_ms": collar,
                                    "skip_overlap": skip,
                                    "compared": False,
                                    "reason": "original md-eval emits no components for this empty input side",
                                    "original_exit_code": process.returncode,
                                    "original_stderr": process.stderr[-1000:],
                                }
                            )
                            break
                        raise RuntimeError(
                            f"original md-eval produced no components: {name}, collar={collar}, skip={skip}: {process.stderr[-1000:]}"
                        )
                    observed[key] = float(values[-1]) * 1000
                    if abs(observed[key] - continuous[key]) > 0.011:
                        mismatch.append(
                            f"DER component {key} differs at collar={collar},skip={skip}"
                        )
                if observed is None:
                    continue
                der_rows.append(
                    {
                        "compared": True,
                        "collar_ms": collar,
                        "skip_overlap": skip,
                        "local": continuous,
                        "md_eval_components": observed,
                    }
                )
            result = {
                "case": name,
                "actual": actual,
                "original_dscore_jer": reference_jer,
                "original_dscore_error": original_error,
                "der_conditions": der_rows,
                "mismatch": mismatch,
                "inputs": {
                    p.name: p.read_text()
                    for p in (work / "ref.rttm", work / "hyp.rttm", work / "all.uem")
                },
            }
            results.append(result)
            if mismatch:
                failures.append(name)

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    report = {
        "schema_version": 1,
        "synthetic_only": True,
        "reference_commit": COMMIT,
        "reference_files_sha256": {
            name: sha(reference_root / name)
            for name in (
                "scorelib/metrics.py",
                "scorelib/score.py",
                "scorelib/turn.py",
                "scorelib/rttm.py",
                "scorelib/uem.py",
                "scorelib/md-eval-22.pl",
            )
        },
        "implementation_sha256": {
            name: sha(ROOT / name)
            for name in (
                "workbench/speaker_scoring_io.py",
                "workbench/diarization_metrics.py",
                "workbench/speaker_benchmark.py",
                "scripts/verify_recording_metrics.py",
            )
        },
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "intervaltree")
        },
        "protocol": "original dscore RTTM/UEM readers, CLI trim/merge, default turns_to_frames(.010), contingency_matrix and jer",
        "jer_comparison_tolerance": 1e-12,
        "der_printed_component_tolerance_ms": 0.011,
        "cases": len(results),
        "failed_cases": failures,
        "results": results,
        "limits": "Software parity only, including original zero-union failure; no human gold or actual model quality",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "cases": len(results),
                "failed_cases": failures,
                "report": str(args.output.resolve()),
            }
        )
    )
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
