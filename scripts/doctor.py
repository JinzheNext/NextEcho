from __future__ import annotations

from workbench.doctor import print_human_report, run_doctor

if __name__ == "__main__":
    report = run_doctor()
    print_human_report(report)
    raise SystemExit(0 if report.ok else 1)
