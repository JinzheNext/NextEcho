"""Verified full frame export for the pinned V3 push/finish-only path.

No preview/flush calls are used. Still-retained exported bytes must remain
identical, and compaction may never skip a frame which was not exported.
"""
from __future__ import annotations

import array
import math
import os
from pathlib import Path
import sys

from .io_utils import file_hash
from .nemotron_timeline import segment_retained_probabilities

STREAM_CONTRACT = "pinned-v3-strict-push-finish-no-preview-v1"


class ProbabilityArchive:
    def __init__(self, path: Path, *, slots=8):
        if sys.byteorder != "little" or slots != 8:
            raise ValueError("native frame export requires a little-endian worker")
        self.path = path
        self.partial = path.with_suffix(path.suffix + ".partial")
        if path.exists():
            raise ValueError("refusing to overwrite a completed probability archive")
        self.file = self.partial.open("xb+")
        self.partial.chmod(0o600)
        self.slots, self.frames, self.updates = slots, 0, 0
        self.overlap_checks = self.overlap_frame_observations = 0

    def update(self, *, first_frame, frame_count, retained_bytes):
        if (type(first_frame) is not int or type(frame_count) is not int
                or not 0 <= first_frame <= self.frames <= frame_count
                or len(retained_bytes) != (frame_count - first_frame) * self.slots * 4):
            raise ValueError("probability export missed frames, regressed or has invalid bytes")
        overlap = (self.frames - first_frame) * self.slots * 4
        self.file.seek(first_frame * self.slots * 4)
        if self.file.read(overlap) != retained_bytes[:overlap]:
            raise ValueError("native probabilities rewrote exported frames; cannot declare a full archive")
        delta = retained_bytes[overlap:]
        values = array.array("f", delta)
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
            raise ValueError("invalid native probability delta")
        self.file.seek(self.frames * self.slots * 4)
        self.file.write(delta)
        self.frames = frame_count
        self.updates += 1
        self.overlap_checks += bool(overlap)
        self.overlap_frame_observations += overlap // (self.slots * 4)

    def finish(self, *, frame_count):
        if self.frames != frame_count or frame_count <= 0:
            raise ValueError("incomplete full probability archive")
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()
        os.link(self.partial, self.path)  # Atomic no-overwrite publication.
        self.partial.unlink()
        return {"path": str(self.path.resolve()), "sha256": file_hash(self.path),
                "format": "little-endian-float32-frame-major-8-channels", "first_frame": 0,
                "frame_count": self.frames, "completed": True, "stream_contract": STREAM_CONTRACT,
                "updates": self.updates, "overlap_checks": self.overlap_checks,
                "overlap_frame_observations": self.overlap_frame_observations}

    def close(self):
        self.file.close()


def verify_full_probability_archive(data, folder):
    archive = data.get("full_probability_archive")
    if (not isinstance(archive, dict) or archive.get("stream_contract") != STREAM_CONTRACT
            or archive.get("completed") is not True or archive.get("first_frame") != 0
            or archive.get("frame_count") != data["frame_count"]
            or archive.get("format") != "little-endian-float32-frame-major-8-channels"
            or type(archive.get("updates")) is not int or archive["updates"] < 1
            or type(archive.get("overlap_checks")) is not int or not 0 <= archive["overlap_checks"] <= archive["updates"]):
        raise ValueError("invalid full probability archive protocol")
    path = Path(archive["path"])
    if (not path.is_absolute() or not path.resolve().is_relative_to(folder.resolve())
            or not path.is_file() or path.stat().st_size != data["frame_count"] * 8 * 4
            or file_hash(path) != archive["sha256"]):
        raise ValueError("full probability archive changed or is incomplete/outside probe")
    values = array.array("f", path.read_bytes())
    if sys.byteorder != "little":
        values.byteswap()
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
        raise ValueError("nonfinite/out-of-range full native probability")
    with path.open("rb") as stream:
        stream.seek(data["retained_frame_start"] * 8 * 4)
        if stream.read() != Path(data["probability_path"]).read_bytes():
            raise ValueError("full archive and final retained tail disagree")
    rebuilt = segment_retained_probabilities(values, first_frame=0, cadence=data["seconds_per_frame"])
    expected = sorted((r["speaker"], r["start"], r["end"]) for r in rebuilt)
    actual = sorted((r["speaker"], r["start"], r["end"]) for r in data["segments"])
    if len(expected) != len(actual) or any(a[0] != b[0] or max(abs(a[1]-b[1]), abs(a[2]-b[2])) > 1e-6
                                            for a, b in zip(expected, actual)):
        raise ValueError("full native probabilities do not reproduce final native segments")
    return {"full_native_frames_verified": data["frame_count"], "native_default_segments_compared": len(actual),
            "retained_tail_bytes_equal": True, "scope": "archive continuity and native default segmentation, not acoustic truth"}
