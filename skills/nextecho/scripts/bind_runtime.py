"""Bind an already installed runtime without modifying models or environments."""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import tempfile
from pathlib import Path


def binding_path():
    if os.environ.get("NEXTECHO_SKILL_BINDING"):
        return Path(os.environ["NEXTECHO_SKILL_BINDING"]).expanduser()
    if platform.system() == "Darwin":
        return Path.home() / "Library/Application Support/NextEcho/skill-binding.json"
    return Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "nextecho/skill-binding.json"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--runtime-dir", type=Path)
    p.add_argument("--binding", type=Path, default=binding_path())
    a = p.parse_args()
    python = a.python.expanduser().absolute()
    checked = subprocess.run([str(python), "-m", "workbench.cli", "--version"],
                             cwd=tempfile.gettempdir(), capture_output=True, text=True, timeout=10)
    if checked.returncode or not checked.stdout.strip().endswith("protocol=1"):
        p.error("解释器中没有兼容NextEcho；请先安装运行时，再绑定")
    version = re.fullmatch(r"nextecho (\d+)\.(\d+)\.(\d+) protocol=1", checked.stdout.strip())
    if not version or tuple(map(int, version.groups())) < (0, 2, 5):
        p.error("此Skill的持久意图、恢复及说话人精修需要NextEcho>=0.2.5；请先升级运行时，再绑定")
    if a.config and not a.config.expanduser().is_file():
        p.error("指定的config不存在")
    target = a.binding.expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = {"schema_version": 1, "python": str(python),
               "runtime_version": ".".join(version.groups()),
               "config": str(a.config.expanduser().absolute()) if a.config else "",
               "runtime_dir": str(a.runtime_dir.expanduser().absolute()) if a.runtime_dir else ""}
    fd, scratch = tempfile.mkstemp(dir=target.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(scratch, target)
    finally:
        Path(scratch).unlink(missing_ok=True)
    print(target)


if __name__ == "__main__":
    main()
