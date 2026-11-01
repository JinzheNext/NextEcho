"""Migrate path-keyed native ASR identities only across identical code bytes."""
from __future__ import annotations

import json
from pathlib import Path

from .io_utils import file_hash

NATIVE_CODE = ("local_asr.py", "local_asr_worker.py", "audio_segments.py", "polish_context.py", "chinese_text.py")


def relocated_native_identity(previous, item_dir, model_identity, worker_python, current_root):
    try:
        old_root = previous.get("cache_implementation_root")
        if not old_root:
            owner = json.loads((item_dir / "logs/asr.guard.owner.json").read_text())
            tail = owner["birth"].split(" " + worker_python + " ", 1)[1]
            if "local_asr_worker.py " not in tail:
                return None
            old_root = str(Path(tail.split("local_asr_worker.py ", 1)[0] + "local_asr_worker.py").parent)
        root = Path(old_root)
        if not root.is_absolute() or root.resolve() == current_root.resolve():
            return None
        identity = dict(model_identity)
        hashes = {}
        for name in NATIVE_CODE:
            current, old = current_root / name, root / name
            if not old.is_file() or file_hash(old) != identity.get(str(current)):
                return None
            hashes[name] = identity.pop(str(current))
            identity[str(old)] = hashes[name]
        return identity, {"kind": "identical-recognition-code-relocated", "from": str(root),
                          "to": str(current_root), "code_sha256": hashes}
    except (OSError, KeyError, IndexError, TypeError, ValueError):
        return None
