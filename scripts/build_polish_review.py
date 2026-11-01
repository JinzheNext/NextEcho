"""Build an offline diff review from private adopted-output records.

Input: {"records": [{"id", "suite", "original", "program", "speaker",
"time_label", "models": {"9B": {"text", "fallback"}}}], "metadata": {...}}.
Audio references, when present, are relative to the output HTML.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from workbench.review import write_review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    records = data['records']
    if len({r['id'] for r in records}) != len(records):
        parser.error('样本ID必须唯一')
    for row in records:
        for field in ('id', 'original', 'suite', 'program', 'speaker', 'time_label'):
            if not isinstance(row.get(field), str):
                parser.error('缺少字符串字段：' + field)
        if not row.get('models') or any(not isinstance(r.get('text'), str) for r in row['models'].values()):
            parser.error('模型结果必须提供实际采用的text，不能以原稿字段代替')
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    write_review(output, records, metadata=data.get('metadata'))
    print(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
