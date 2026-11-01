from __future__ import annotations

import argparse
import json
from pathlib import Path


def _enable_embedded_quantization_compat() -> None:
    """Teach mlx-qwen3-asr 0.3.x to read community quantization metadata.

    Some MLX Qwen checkpoints (including the 5-bit community build) keep the
    ``bits``/``group_size`` declaration inside ``config.json`` instead of a
    separate ``quantization_config.json``.  The runtime otherwise defaults to
    4-bit when it sees ``.scales`` tensors, producing a misleading shape
    mismatch before inference starts.  This small compatibility shim is kept
    inside the isolated worker so the pinned third-party package is untouched.
    """
    try:
        from mlx_qwen3_asr import load_models
    except ImportError:
        return

    original = load_models._read_quantization_config

    def read_quantization_config(model_path):
        config = original(model_path)
        if config is not None:
            return config
        config_path = Path(model_path) / "config.json"
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        embedded = raw.get("quantization_config")
        return embedded if isinstance(embedded, dict) else None

    load_models._read_quantization_config = read_quantization_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated Qwen3-ASR MLX worker")
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--language", default="auto")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    _enable_embedded_quantization_compat()
    from mlx_qwen3_asr import transcribe  # Imported only in the child process.
    from .asr import qwen_runtime_version

    language = None if args.language in {"", "auto"} else {"zh": "Chinese", "en": "English"}.get(args.language, args.language)
    result = transcribe(
        str(args.audio),
        model=args.model,
        language=language,
        return_timestamps=True,
    )
    args.output.write_text(
        json.dumps(
            {
                "text": result.text,
                "language": result.language,
                "words": result.segments or [],
                "model": args.model,
                "version": qwen_runtime_version(),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
