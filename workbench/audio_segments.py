"""Checked global time windows and conservative overlap merging."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioWindow:
    start_sample: int
    end_sample: int
    sample_rate: int = 16000

    def __post_init__(self):
        if self.sample_rate <= 0 or self.start_sample < 0 or self.end_sample <= self.start_sample:
            raise ValueError("Invalid audio window")

    @property
    def start_ms(self):
        return round(self.start_sample * 1000 / self.sample_rate)

    @property
    def end_ms(self):
        return round(self.end_sample * 1000 / self.sample_rate)


def split_window(start: int, end: int, *, sample_rate: int = 16000, max_seconds: int = 30):
    step = sample_rate * max_seconds
    if step <= 0 or end < start:
        raise ValueError("Invalid window limits")
    return [AudioWindow(left, min(left + step, end), sample_rate) for left in range(start, end, step)]


def merge_recognized_segments(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    result, review = [], []
    last_start = -1
    for row in rows:
        row = dict(row)
        start, end = int(row["start_ms"]), int(row["end_ms"])
        if start < last_start or start < 0 or end <= start:
            raise ValueError("Non-monotonic or empty audio time window")
        last_start = start
        if not str(row.get("text", "")).strip():
            continue
        row["raw_start_ms"], row["raw_end_ms"] = start, end
        row["start_ms"] = int(row.get("retain_start_ms", start))
        row["end_ms"] = int(row.get("retain_end_ms", end))
        if result and start < result[-1]["raw_end_ms"]:
            before = "".join(c for c in result[-1]["text"] if c.isalnum())
            positions = [i for i, c in enumerate(row["text"]) if c.isalnum()]
            text = "".join(row["text"][i] for i in positions)
            matches = [n for n in range(4, min(len(before), len(text), 120) + 1) if before[-n:] == text[:n]]
            if matches:
                count = max(matches)
                if count == len(text) and end > result[-1]["raw_end_ms"]:
                    review.append({"reason": "ambiguous_full_repetition", "start_ms": start})
                else:
                    row["text"] = row["text"][positions[count - 1] + 1:].lstrip(" ，。！？、；：,.!?;: \n")
                    row["overlap_chars_removed"] = count
            else:
                review.append({"reason": "ambiguous_overlap", "start_ms": start})
            row["start_ms"] = max(row["start_ms"], result[-1]["end_ms"])
        if row["text"].strip() and row["end_ms"] > row["start_ms"]:
            row["id"] = f"seg_{len(result):05d}"
            result.append(row)
    return result, review
