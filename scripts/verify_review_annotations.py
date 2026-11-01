"""Check exported author decisions against the exact generated review dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from workbench.review_contract import validate_annotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.page.stat().st_size > 64 * 1024**2 or args.annotations.stat().st_size > 16 * 1024**2:
            raise ValueError("审核文件超过校验大小上限")
        matches = re.findall(r'<script id="(?:identity|review)-data" type="application/json">(.*?)</script>', args.page.read_text(), re.S)
        if len(matches) != 1:
            raise ValueError("页面缺少唯一审核数据集")
        dataset = json.loads(matches[0])
        export = validate_annotations(json.loads(args.annotations.read_text()), dataset)
        field = "verdict" if dataset["review_kind"] == "identity" else "meaning"
        judged = sum(row["decision"][field] != "pending" for row in export["items"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error("审核版本/来源/字段校验失败：" + str(exc)[:200])
    print(json.dumps({"status": "annotation-schema-verified", "review_kind": export["review_kind"],
        "dataset_fingerprint": dataset["fingerprint"], "items": len(export["items"]), "judged": judged,
        "pending": len(export["items"]) - judged, "human_opinion_quality": "not-automatically-verified"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
