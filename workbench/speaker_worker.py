"""Isolated existing speaker algorithm, with a read-only metadata probe."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from workbench.config import NextEchoConfig
from workbench.io_utils import redact_structure, write_json
from workbench.speaker_transcript import build_speaker_transcript, detect_speaker_backend_status


def main():
    request = json.loads(Path(sys.argv[1]).read_text())
    config = NextEchoConfig(request["config"])
    if "--probe" in sys.argv:
        write_json(Path(request["output"]), detect_speaker_backend_status(config=config))
        return
    if config.get_path("asr", "strict_selection", default=False):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    kwargs = request["kwargs"]
    if kwargs.get("output_dir"):
        kwargs["output_dir"] = Path(kwargs["output_dir"])
    def progress(event):
        write_json(Path(request["progress"]), redact_structure(event))
    try:
        result = build_speaker_transcript(request["input"], config=config, progress_callback=progress, **kwargs)
        write_json(Path(request["output"]), {"schema_version": 1, "result": result})
    except Exception as exc:
        write_json(Path(request["output"]), {"schema_version": 1, "error_code": getattr(exc, "code", "SPEAKER_FAILED"),
                   "message": str(exc), "blocked": getattr(exc, "blocked", False)})


if __name__ == "__main__":
    main()
