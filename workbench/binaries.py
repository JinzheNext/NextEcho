from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def _candidate_bin_dirs() -> list[Path]:
    project_root = Path(__file__).resolve().parent.parent
    venv_bin = project_root / ".venv" / ("Scripts" if os.name == "nt" else "bin")
    current_python_bin = Path(sys.executable).resolve().parent
    candidates = [current_python_bin, venv_bin]
    seen: set[Path] = set()
    unique: list[Path] = []
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def find_binary(name: str) -> str | None:
    for directory in _candidate_bin_dirs():
        found = shutil.which(name, path=str(directory))
        if found:
            return found
    return shutil.which(name)
