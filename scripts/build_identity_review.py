"""Generate a private offline identity review without loading a model."""
import argparse
from pathlib import Path

from workbench.identity_review import write_identity_review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dialogue", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--semantic-review", type=Path)
    parser.add_argument("--audio-origin-ms", type=int, required=True, help="试听WAV的整集起点；完整原音频显式填0")
    parser.add_argument("--audio-sha256", help="独立来源记录的预期hash；默认使用dialogue.audio_sha256")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    write_identity_review(output, args.dialogue.expanduser().resolve(), args.audio.expanduser().resolve(),
                          semantic_review=args.semantic_review.expanduser().resolve() if args.semantic_review else None,
                          audio_origin_ms=args.audio_origin_ms, expected_audio_sha256=args.audio_sha256)
    print(output)


if __name__ == "__main__":
    main()
