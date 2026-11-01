"""Exact recording-wide character error metrics; no model or native dependency.

cpCER is the character-unit adaptation of concatenated minimum-permutation
WER. Fixed-map speaker CER is labelled separately, never inferred from cpCER
and never presented as NIST's multi-dimensional SpSub/asclite metric.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import unicodedata

from .speaker_benchmark import _maximum_weight_pairs

NORMALIZATION = "nfkc-no-whitespace-v1"
MAX_CHARS = 200_000


def normalize_characters(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("transcript text must be a string")
    # Keep punctuation, decimal separators, script and case. No ITN,
    # simplified/traditional conversion, spelling correction or filler deletion.
    return "".join(char for char in unicodedata.normalize("NFKC", text) if not char.isspace())


def edit_distance(reference: str, hypothesis: str) -> int:
    """Exact unit-cost Levenshtein using a bounded-width bit vector."""
    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        raise ValueError("edit distance requires two strings")
    if max(len(reference), len(hypothesis)) > MAX_CHARS:
        raise ValueError("character scoring supports at most 200000 characters per stream")
    if len(reference) > len(hypothesis):
        reference, hypothesis = hypothesis, reference
    width = len(reference)
    if not width:
        return len(hypothesis)
    masks = defaultdict(int)
    for index, char in enumerate(reference):
        masks[char] |= 1 << index
    mask, last = (1 << width) - 1, 1 << (width - 1)
    positive, negative, score = mask, 0, width
    for char in hypothesis:
        match = masks.get(char, 0)
        diagonal = (((match & positive) + positive) ^ positive) | match | negative
        horizontal_negative = positive & diagonal
        horizontal_positive = negative | ~(positive | diagonal)
        score += bool(horizontal_positive & last) - bool(horizontal_negative & last)
        shifted = (horizontal_positive << 1) | 1
        negative = (shifted & diagonal) & mask
        positive = ((horizontal_negative << 1) | ~(shifted | diagonal)) & mask
    return score


def _ratio(errors, reference_chars, hypothesis_chars):
    return {"errors": errors, "reference_chars": reference_chars, "hypothesis_chars": hypothesis_chars,
            "error_rate": errors / reference_chars if reference_chars else None,
            "denominator_defined": bool(reference_chars)}


def score_characters(reference, hypothesis, *, fixed_mapping=None, mapping_source=None):
    """Rows are already ordered and UEM-selected; speaker=None is unattributed.

    Missing/extra streams are scored against empty strings. Every real
    hypothesis character remains in the denominator diagnostics and costs.
    """
    ref_text, hyp_text = [], []
    refs, hyps = {}, {}
    for rows, streams, all_text, kind in ((reference, refs, ref_text, "reference"),
                                         (hypothesis, hyps, hyp_text, "hypothesis")):
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("text segments must be objects")
            label, text = row.get("speaker_id"), normalize_characters(row.get("text"))
            if label is not None and (not isinstance(label, str) or not label.strip()):
                raise ValueError("speaker_id must be a nonempty string or null")
            if kind == "reference" and label is None:
                raise ValueError("reference speaker IDs are required; unknown gold is not silence")
            streams[label] = streams.get(label, "") + text
            all_text.append(text)
        if len(streams) > 256 or sum(map(len, all_text)) > MAX_CHARS:
            raise ValueError("recording scoring supports <=256 streams and <=200000 normalized characters")
    full_ref, full_hyp = "".join(ref_text), "".join(hyp_text)
    cer = _ratio(edit_distance(full_ref, full_hyp), len(full_ref), len(full_hyp))
    ref_labels = sorted(refs)
    hyp_labels = sorted(hyps, key=lambda value: (value is None, value or ""))
    size = max(len(ref_labels), len(hyp_labels))
    costs = [[edit_distance(refs[ref_labels[r]] if r < len(ref_labels) else "",
                            hyps[hyp_labels[h]] if h < len(hyp_labels) else "")
              for h in range(size)] for r in range(size)]
    pairs = _maximum_weight_pairs([[-value for value in row] for row in costs])
    assignments = [{"reference_speaker": ref_labels[r] if r < len(ref_labels) else None,
                    "hypothesis_speaker": hyp_labels[h] if h < len(hyp_labels) else None,
                    "reference_missing": r >= len(ref_labels), "hypothesis_missing": h >= len(hyp_labels),
                    "hypothesis_unattributed": h < len(hyp_labels) and hyp_labels[h] is None,
                    "errors": costs[r][h]} for r, h in pairs]
    cp = {**_ratio(sum(costs[r][h] for r, h in pairs), len(full_ref), len(full_hyp)),
          "assignment": assignments, "mapping_scope": "recording", "mapping_objective": "minimum_character_edit_errors",
          "assignment_is_identity_evidence": False, "reference_streams": len(refs), "hypothesis_streams": len(hyps)}
    fixed = None
    if fixed_mapping is not None:
        if not isinstance(fixed_mapping, dict) or not isinstance(mapping_source, str) or mapping_source not in {
            "declared-registered-identities", "human-audio-review", "time-overlap-oracle"
        }:
            raise ValueError("fixed mapping requires an explicit declared source")
        if (any(not isinstance(key, str) or not isinstance(value, str)
                or key not in hyps or value not in refs for key, value in fixed_mapping.items())
                or len(set(fixed_mapping.values())) != len(fixed_mapping)):
            raise ValueError("fixed mapping must be one-to-one between actual streams; never map unattributed text")
        inverse = {value: key for key, value in fixed_mapping.items()}
        errors = sum(edit_distance(refs[label], hyps[inverse[label]] if label in inverse else "") for label in refs)
        errors += sum(len(text) for label, text in hyps.items() if label not in fixed_mapping)
        fixed = {**_ratio(errors, len(full_ref), len(full_hyp)), "metric_protocol": "sa-cer-fixed-map-v1",
                 "mapping_source": mapping_source, "mapping": fixed_mapping,
                 "unmapped_hypothesis_chars": sum(len(text) for label, text in hyps.items() if label not in fixed_mapping),
                 "scope": "per-speaker character edits under supplied fixed mapping; not NIST SpSub/asclite"}
    return {"normalization": NORMALIZATION, "cer": cer, "cpcer": cp,
            "sa_cer_fixed_map": fixed,
            "sa_cer_status": "scored_with_declared_mapping" if fixed is not None else "not_scored_no_fixed_mapping",
            "unattributed_hypothesis_chars": len(hyps.get(None, "")), "model_release_approved": False}


def _json_file(path):
    if not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("text scoring input must be a regular JSON file <=32MiB")
    content = path.read_bytes()
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate JSON key in text scoring input")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("nonfinite JSON value in text scoring input")
    return json.loads(content, object_pairs_hook=pairs, parse_constant=invalid), hashlib.sha256(content).hexdigest()


def evaluate_text_files(reference, hypothesis, *, root, recording_id, source_audio_sha256,
                        uem_ms, duration_ms, task="raw-asr", fixed_mapping=None,
                        channel="1", reference_speakers=None, hypothesis_speakers=None):
    """Validate frozen text layers, never crop a crossing utterance by letters."""
    if not isinstance(task, str) or task not in {"raw-asr", "postprocessed"}:
        raise ValueError("text task must be raw-asr or postprocessed")
    def path(value):
        if not isinstance(value, str) or not value:
            raise ValueError("text scoring path is required")
        p = Path(value).expanduser()
        return p.resolve() if p.is_absolute() else (Path(root)/p).resolve()
    windows = []
    for start, end in sorted(uem_ms):
        if windows and start <= windows[-1][1]:
            windows[-1][1] = max(end, windows[-1][1])
        else:
            windows.append([start, end])
    def load(value, kind):
        if not isinstance(value, dict) or value.get("annotation_kind") != kind:
            raise ValueError("explicit text annotation declaration required")
        if kind == "human-verbatim" and value.get("annotation_status") != "complete":
            raise ValueError("human text reference is pending; cannot compute CER")
        file_path = path(value.get("path"))
        payload, digest = _json_file(file_path)
        if (not isinstance(payload, dict) or payload.get("schema_version") != 1
                or payload.get("annotation_kind") != kind or payload.get("recording_id") != recording_id
                or payload.get("source_audio_sha256") != source_audio_sha256
                or str(payload.get("channel", "1")) != channel):
            raise ValueError("text annotation must bind kind/recording and actual source audio hash")
        if kind == "human-verbatim" and payload.get("annotation_status") != "complete":
            raise ValueError("human text file is incomplete")
        layer = payload.get("text_layer")
        if kind == "human-verbatim" and layer != "verbatim-reference":
            raise ValueError("reference must be human verbatim text")
        if kind == "hypothesis-transcript" and layer not in (
            {"raw-asr", "speaker-attributed-raw-asr"} if task == "raw-asr" else {"postprocessed"}
        ):
            raise ValueError("text layer/task mismatch; enhanced text cannot be scored as raw ASR")
        rows = payload.get("segments")
        if not isinstance(rows, list):
            raise ValueError("text annotation segments are required")
        selected, excluded, ids, last = [], [], set(), -1.0
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"] or row["id"] in ids:
                raise ValueError("text segments require unique IDs")
            ids.add(row["id"])
            start, end = row.get("start_ms"), row.get("end_ms")
            if (type(start) not in (int, float) or type(end) not in (int, float)
                    or not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= duration_ms + 1e-6
                    or start < last or not isinstance(row.get("text"), str)):
                raise ValueError("text rows require ordered, finite measured ranges within source audio")
            label = row.get("speaker_id")
            if (label is not None and (not isinstance(label, str) or not label.strip())) or (
                kind == "human-verbatim" and label is None
            ):
                raise ValueError("text rows require valid speaker IDs; only hypotheses may be unattributed")
            last = start
            overlaps = any(start < b and end > a for a, b in windows)
            contained = any(a <= start and end <= b for a, b in windows)
            if overlaps and not contained:
                raise ValueError("text segment crosses UEM; need complete clip text or real measured split, never proportional cropping")
            (selected if contained else excluded).append(row)
        if (not any(normalize_characters(row["text"]) for row in rows)) != (payload.get("content_state") == "no_speech"):
            raise ValueError("empty text requires explicit no_speech; pending is not silence")
        source_evidence = None
        if kind == "hypothesis-transcript":
            source = payload.get("text_source")
            if not isinstance(source, dict):
                raise ValueError("hypothesis requires a frozen actual producer text_source")
            original, original_hash = _json_file(path(source.get("path")))
            field = "transcription" if task == "raw-asr" else "enhanced_transcription"
            if source.get("sha256") != original_hash or source.get("field") != field:
                raise ValueError("producer text layer/hash mismatch")
            original_rows = original.get(field) if isinstance(original, dict) else None
            if not isinstance(original_rows, list) or any(not isinstance(r, dict) or not isinstance(r.get("text"), str) for r in original_rows):
                raise ValueError("producer is missing its declared actual text layer")
            if "".join(row["text"] for row in rows) != "".join(row["text"] for row in original_rows):
                raise ValueError("scoring annotation text differs from frozen producer; no correction/reordering allowed")
            if source.get("timing_policy") != "producer-segment-envelopes-v1" or len(rows) != len(original_rows):
                raise ValueError("text timing requires original whole producer segments; fabricated splits are not supported")
            for row, original_row in zip(rows, original_rows):
                offsets = original_row.get("offsets")
                if (not isinstance(offsets, dict) or row["id"] != original_row.get("id")
                        or row["text"] != original_row["text"]
                        or type(offsets.get("from")) not in (int, float)
                        or type(offsets.get("to")) not in (int, float)
                        or row["start_ms"] != offsets["from"] or row["end_ms"] != offsets["to"]):
                    raise ValueError("text segment identity/text/timing differs from frozen producer")
            source_evidence = {"sha256": original_hash, "field": field,
                               "timing_policy": source["timing_policy"],
                               "producer_timing_precision": original.get("timing_precision"),
                               "timing_accuracy_verified": False}
        labels = reference_speakers if kind == "human-verbatim" else hypothesis_speakers
        if labels is not None and any(row.get("speaker_id") not in labels and not (
            kind == "hypothesis-transcript" and row.get("speaker_id") is None
        ) for row in selected):
            raise ValueError("text speaker IDs must match the scored recording's acoustic labels")
        return selected, payload, {"sha256": digest, "text_layer": layer,
            "selected_segment_ids": [r["id"] for r in selected], "excluded_segment_ids": [r["id"] for r in excluded],
            "producer": source_evidence}
    ref_rows, ref_payload, ref_evidence = load(reference, "human-verbatim")
    hyp_rows, _, hyp_evidence = load(hypothesis, "hypothesis-transcript")
    mapping, mapping_source, mapping_sha = None, None, None
    if fixed_mapping is not None:
        if not isinstance(fixed_mapping, dict):
            raise ValueError("fixed mapping requires a declared JSON file")
        data, mapping_sha = _json_file(path(fixed_mapping.get("path")))
        if (not isinstance(data, dict) or data.get("schema_version") != 1 or data.get("annotation_kind") != "fixed-speaker-mapping"
                or data.get("annotation_status") != "complete" or data.get("recording_id") != recording_id
                or data.get("source_audio_sha256") != source_audio_sha256
                or str(data.get("channel", "1")) != channel):
            raise ValueError("fixed mapping must be complete and bind recording/audio")
        mapping, mapping_source = data.get("mapping"), data.get("mapping_source")
        if not isinstance(mapping, dict):
            raise ValueError("fixed mapping dictionary is required")
    scores = score_characters(ref_rows, hyp_rows, fixed_mapping=mapping, mapping_source=mapping_source)
    tags = ("short_answer", "overlap", "fragment", "boundary")
    strata = {"status": "not_annotated", "definitions": None, "denominators": None}
    if ref_payload.get("strata_status") == "complete":
        definitions = ref_payload.get("strata_definitions")
        if not isinstance(definitions, dict) or set(definitions) != set(tags) or any(not isinstance(x, str) or not x.strip() for x in definitions.values()):
            raise ValueError("complete reference strata require all four frozen definitions")
        counts = {tag: {"reference_segments": 0, "reference_chars": 0, "reference_speaker_ms": 0} for tag in tags}
        durations = {tag: defaultdict(list) for tag in tags}
        for row in ref_rows:
            assigned = row.get("tags")
            if not isinstance(assigned, list) or any(not isinstance(tag, str) or tag not in tags for tag in assigned) or len(set(assigned)) != len(assigned):
                raise ValueError("complete strata require an explicit valid tags list on every selected reference segment")
            for tag in assigned:
                counts[tag]["reference_segments"] += 1
                counts[tag]["reference_chars"] += len(normalize_characters(row["text"]))
                durations[tag][row["speaker_id"]].append((row["start_ms"], row["end_ms"]))
        for tag, speakers in durations.items():
            for ranges in speakers.values():
                merged = []
                for start, end in sorted(ranges):
                    if merged and start <= merged[-1][1]:
                        merged[-1][1] = max(end, merged[-1][1])
                    else:
                        merged.append([start, end])
                counts[tag]["reference_speaker_ms"] += sum(b - a for a, b in merged)
        strata = {"status": "complete", "definitions": definitions, "denominators": counts,
                  "scope": "reference-only diagnostics; categories may overlap; not stratum error rates"}
    return {**scores, "task": task, "reference": ref_evidence, "hypothesis": hyp_evidence,
            "fixed_mapping_sha256": mapping_sha, "reference_strata": strata}


def evaluate_character_span_attribution(hypothesis_path, reference_turns, *, root,
                                        recording_id, source_audio_sha256,
                                        uem_ms, duration_ms):
    """Score native-timed character spans against one recording's gold turns.

    This is an acoustic-speaker agreement metric, not CER and not a name
    resolver. Review-envelope-only characters remain visible in the coverage
    diagnostics but cannot enter the score.
    """
    base = Path(root).resolve()
    candidate = Path(hypothesis_path).expanduser()
    candidate = candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
    payload, artifact_hash = _json_file(candidate)
    if (not isinstance(payload, dict) or payload.get("schema_version") != 2
            or payload.get("annotation_kind") != "speaker-attributed-character-spans"
            or payload.get("status") not in {"ready", "partial"}
            or payload.get("recording_id") != recording_id
            or payload.get("source_audio_sha256") != source_audio_sha256
            or payload.get("text_reconstruction_exact") is not True):
        raise ValueError("character-span artifact must bind this recording/audio and preserve exact text")

    def resolve_path(value):
        if not isinstance(value, str) or not value:
            raise ValueError("character-span source path is required")
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (base / path).resolve()

    def digest_file(path):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    source_path = resolve_path(payload.get("source_path"))
    if not source_path.is_file() or digest_file(source_path) != payload.get("source_sha256"):
        raise ValueError("character-span text source hash mismatch")
    audio_value = payload.get("source_audio")
    if audio_value:
        audio_path = resolve_path(audio_value)
        if not audio_path.is_file() or digest_file(audio_path) != source_audio_sha256:
            raise ValueError("character-span source audio hash mismatch")
    elif payload.get("source_audio_sha256") != source_audio_sha256:
        raise ValueError("character-span source audio binding is missing")
    original, original_hash = _json_file(source_path)
    if payload.get("source_field") != "enhanced_transcription":
        raise ValueError("character-span scoring requires the frozen enhanced text layer")
    source_rows = original.get("enhanced_transcription") if isinstance(original, dict) else None
    if not isinstance(source_rows, list):
        raise ValueError("character-span source is missing enhanced_transcription")
    originals = {}
    total_chars = 0
    for index, source_row in enumerate(source_rows):
        if not isinstance(source_row, dict) or not isinstance(source_row.get("text"), str):
            raise ValueError("invalid producer segment in character-span source")
        segment_id = str(source_row.get("id") or f"seg_{index:05d}")
        if segment_id in originals:
            raise ValueError("duplicate producer segment ID in character-span source")
        source_text = source_row["text"]
        offsets = source_row.get("offsets") or {}
        originals[segment_id] = {"text": source_text,
            "sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
            "start_ms": offsets.get("from"), "end_ms": offsets.get("to"), "spans": []}
        total_chars += len(normalize_characters(source_text))

    rows = payload.get("transcription")
    if not isinstance(rows, list):
        raise ValueError("character-span transcription rows are required")
    ids = set()
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]
                or row["id"] in ids or not isinstance(row.get("text"), str)):
            raise ValueError("character-span rows require unique IDs and exact source text")
        ids.add(row["id"])
        parent = row.get("parent_segment_id")
        source = originals.get(parent)
        start, end = row.get("text_start"), row.get("text_end")
        if (source is None or type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(source["text"])
                or row["text"] != source["text"][start:end]
                or row.get("source_text_sha256") != source["sha256"]):
            raise ValueError("character span does not match its frozen producer text and hash")
        speaker_id = row.get("speaker_id") or None
        if speaker_id is not None and (not isinstance(speaker_id, str) or not speaker_id.strip()):
            raise ValueError("speaker_id must be a nonempty string or null")
        source["spans"].append((start, end, row))

    for segment_id, source in originals.items():
        cursor = 0
        for start, end, row in sorted(source["spans"], key=lambda item: (item[0], item[1])):
            if start != cursor:
                raise ValueError(f"character spans have a gap or overlap in producer segment {segment_id}")
            cursor = end
            voice = row.get("voice_evidence")
            if not isinstance(voice, list):
                raise ValueError("each speaker span requires explicit native timing evidence")
            voice_cursor = start
            for evidence in voice:
                if not isinstance(evidence, dict):
                    raise ValueError("invalid voice evidence span")
                a, b = evidence.get("text_start"), evidence.get("text_end")
                if (type(a) is not int or type(b) is not int or a != voice_cursor
                        or b <= a or b > end or evidence.get("source_text_sha256") != source["sha256"]):
                    raise ValueError("voice evidence must exactly partition its speaker text span")
                voice_cursor = b
                timing_kind = evidence.get("timing_kind")
                if timing_kind not in {"native_interval", "native_onset", "invalid_zero_width", "review_only"}:
                    raise ValueError("unknown character timing evidence kind")
                if timing_kind in {"native_interval", "native_onset", "invalid_zero_width"}:
                    t0 = evidence.get("native_start_ms")
                    t1 = evidence.get("native_end_ms")
                    if type(t0) not in (int, float) or not math.isfinite(t0) or not 0 <= t0 <= duration_ms:
                        raise ValueError("native character evidence has an invalid start")
                    if timing_kind == "native_interval" and (type(t1) not in (int, float) or not math.isfinite(t1) or not t0 < t1 <= duration_ms):
                        raise ValueError("native interval evidence must have a positive measured duration")
                    if timing_kind == "native_onset" and t1 is not None:
                        raise ValueError("native onset evidence cannot claim an end time")
                    if timing_kind == "invalid_zero_width" and (type(t1) not in (int, float) or t1 != t0):
                        raise ValueError("zero-width evidence must preserve its failed interval")
            if voice_cursor != end:
                raise ValueError("voice evidence does not cover the entire speaker text span")
        if source["text"] and cursor != len(source["text"]):
            raise ValueError(f"character spans do not reconstruct producer segment {segment_id}")

    windows = []
    for item in uem_ms:
        if (not isinstance(item, (list, tuple)) or len(item) != 2
                or type(item[0]) not in (int, float) or type(item[1]) not in (int, float)
                or not 0 <= item[0] < item[1] <= duration_ms):
            raise ValueError("invalid UEM window for character-span scoring")
        windows.append([item[0], item[1]])
    windows.sort()
    merged_windows = []
    for start, end in windows:
        if merged_windows and start <= merged_windows[-1][1]:
            merged_windows[-1][1] = max(end, merged_windows[-1][1])
        else:
            merged_windows.append([start, end])

    gold = []
    for turn in reference_turns:
        if not isinstance(turn, dict):
            raise ValueError("reference speaker turns must be objects")
        label, start, end = turn.get("speaker_id"), turn.get("start_ms"), turn.get("end_ms")
        if (not isinstance(label, str) or not label.strip() or type(start) not in (int, float)
                or type(end) not in (int, float) or not math.isfinite(start) or not math.isfinite(end)
                or not 0 <= start < end <= duration_ms):
            raise ValueError("reference speaker turns require valid speaker IDs and positive ranges")
        gold.append({"speaker_id": label, "start_ms": start, "end_ms": end,
                     "overlap": bool(turn.get("overlap"))})

    def point_gold(point):
        active = [turn for turn in gold if turn["start_ms"] <= point < turn["end_ms"]]
        labels = {turn["speaker_id"] for turn in active}
        if len(labels) == 1 and not any(turn["overlap"] for turn in active):
            return next(iter(labels))
        return None

    def interval_gold(start, end):
        relevant = [turn for turn in gold if start < turn["end_ms"] and end > turn["start_ms"]]
        points = sorted({start, end, *(max(start, turn["start_ms"]) for turn in relevant),
                         *(min(end, turn["end_ms"]) for turn in relevant)})
        labels, covered, last = set(), 0.0, start
        for left, right in zip(points, points[1:]):
            if right <= left:
                continue
            active = [turn for turn in relevant if turn["start_ms"] < right and turn["end_ms"] > left]
            current = {turn["speaker_id"] for turn in active}
            if len(current) != 1 or any(turn["overlap"] for turn in active):
                return None
            labels.update(current)
            covered += max(0.0, right - max(last, left))
            last = max(last, right)
        return next(iter(labels)) if covered >= end - start - 1e-6 and len(labels) == 1 else None

    confusion = defaultdict(lambda: defaultdict(int))
    counts = {"native_aligned_chars": 0, "scored_chars": 0, "unattributed_chars": 0,
              "review_only_chars": 0, "invalid_zero_width_chars": 0,
              "outside_uem_chars": 0, "ambiguous_or_silent_gold_chars": 0}
    for source in originals.values():
        for _, _, row in source["spans"]:
            predicted = row.get("speaker_id") or None
            for evidence in row["voice_evidence"]:
                a, b = evidence["text_start"], evidence["text_end"]
                chars = len(normalize_characters(source["text"][a:b]))
                kind = evidence["timing_kind"]
                if kind == "review_only":
                    counts["review_only_chars"] += chars
                    continue
                if kind == "invalid_zero_width":
                    counts["invalid_zero_width_chars"] += chars
                    continue
                t0, t1 = evidence["native_start_ms"], evidence.get("native_end_ms")
                if kind == "native_interval":
                    overlaps_uem = any(t0 < end and t1 > start for start, end in merged_windows)
                    in_uem = any(start <= t0 and t1 <= end for start, end in merged_windows)
                    if overlaps_uem and not in_uem:
                        raise ValueError("native character interval crosses UEM; text cannot be proportionally cropped")
                    if not in_uem:
                        counts["outside_uem_chars"] += chars
                        continue
                    counts["native_aligned_chars"] += chars
                    label = interval_gold(t0, t1)
                else:
                    in_uem = any(start <= t0 < end for start, end in merged_windows)
                    if not in_uem:
                        counts["outside_uem_chars"] += chars
                        continue
                    counts["native_aligned_chars"] += chars
                    label = point_gold(t0)
                if label is None:
                    counts["ambiguous_or_silent_gold_chars"] += chars
                    continue
                counts["scored_chars"] += chars
                if predicted is None:
                    counts["unattributed_chars"] += chars
                else:
                    confusion[predicted][label] += chars

    predicted_labels, gold_labels = sorted(confusion), sorted({label for row in confusion.values() for label in row})
    size = max(len(predicted_labels), len(gold_labels))
    pairs = _maximum_weight_pairs([[confusion[predicted_labels[i]][gold_labels[j]]
                                    if i < len(predicted_labels) and j < len(gold_labels) else 0
                                    for j in range(size)] for i in range(size)]) if size else []
    mapping = {predicted_labels[i]: gold_labels[j] for i, j in pairs
               if i < len(predicted_labels) and j < len(gold_labels)}
    correct = sum(confusion[pred][ref] for pred, ref in mapping.items())
    counts["producer_chars"] = total_chars
    counts["native_character_coverage"] = counts["native_aligned_chars"] / total_chars if total_chars else None
    counts["speaker_accuracy"] = correct / counts["scored_chars"] if counts["scored_chars"] else None
    return {"metric_protocol": "speaker-character-agreement-v1", "recording_id": recording_id,
            "source_audio_sha256": source_audio_sha256, "hypothesis_artifact_sha256": artifact_hash,
            "reference_mapping": mapping, "mapping_scope": "one recording-wide permutation; not identity evidence",
            "counts": counts, "scoring_eligible": bool(counts["scored_chars"]),
            "model_release_approved": False}


def aggregate_text_scores(results):
    """Micro-average only complete common conditions; no partial denominator."""
    if not results:
        return None
    if len({(r["task"], r["normalization"]) for r in results}) != 1:
        raise ValueError("text aggregation requires the same task and normalization")
    def summed(key):
        values = [r[key] for r in results]
        return _ratio(*(sum(v[field] for v in values) for field in (
            "errors", "reference_chars", "hypothesis_chars")))
    fixed = [r["sa_cer_fixed_map"] for r in results]
    sources = {x["mapping_source"] for x in fixed if x is not None}
    fixed_complete = all(x is not None for x in fixed) and len(sources) == 1
    strata = [r["reference_strata"] for r in results]
    all_strata = all(s["status"] == "complete" for s in strata)
    if all_strata and any(s["definitions"] != strata[0]["definitions"] for s in strata):
        raise ValueError("reference strata definitions must remain frozen across recordings")
    return {
        "task": results[0]["task"], "normalization": results[0]["normalization"],
        "recordings": len(results), "cer": summed("cer"), "cpcer": summed("cpcer"),
        "sa_cer_fixed_map": {**summed("sa_cer_fixed_map"), "mapping_source": next(iter(sources)),
                             "metric_protocol": "sa-cer-fixed-map-v1"} if fixed_complete else None,
        "sa_cer_status": "scored_with_common_mapping_source" if fixed_complete else "not_scored_missing_or_mixed_mapping_sources",
        "unattributed_hypothesis_chars": sum(r["unattributed_hypothesis_chars"] for r in results),
        "reference_strata": {"status": "complete", "definitions": strata[0]["definitions"],
            "denominators": {tag: {field: sum(s["denominators"][tag][field] for s in strata)
                                   for field in strata[0]["denominators"][tag]}
                             for tag in strata[0]["denominators"]}} if all_strata else {
                                 "status": "not_scored_incomplete_reference_strata", "denominators": None},
        "mapping_scope": "one assignment per recording, never per UEM window",
        "model_release_approved": False,
    }
