"""Preserve source text while separating native anchors from review envelopes."""
from __future__ import annotations

import difflib
import hashlib
import math
import unicodedata


def _voice_at(point, turns):
    """Native onsets are half-open points, never +/-1ms intervals."""
    active = [r for r in turns if r['start'] * 1000 <= point < r['end'] * 1000]
    labels = {r['speaker'] for r in active}
    if len(labels) > 1 or any(r.get('overlap') for r in active):
        return _overlap(active), 'acoustic_overlap'
    if not active:
        return '归属待核对', 'no_acoustic_activity'
    if any(r.get('uncertain') for r in active):
        return '归属待核对', 'uncertain_voice'
    return next(iter(labels)), 'native_onset'


def _overlap(turns):
    labels = {r['speaker'] for r in turns}
    if len(labels) < 2 or any(r.get('uncertain') for r in turns):
        return '交叠发言（参与声音待核对）'
    return '交叠发言（' + '＋'.join(sorted(labels)) + '）'


def _voice_between(start, end, turns):
    if end <= start:
        return '归属待核对', 'missing_duration'
    relevant = [r for r in turns if min(end, r['end'] * 1000) > max(start, r['start'] * 1000)]
    boundaries = sorted({start, end, *(max(start, r['start'] * 1000) for r in relevant),
                         *(min(end, r['end'] * 1000) for r in relevant)})
    labels, overlap, gap, uncertain = set(), [], False, False
    for left, right in zip(boundaries, boundaries[1:]):
        if right <= left:
            continue
        active = [r for r in relevant if r['start'] * 1000 < right and r['end'] * 1000 > left]
        current = {r['speaker'] for r in active}
        labels.update(current)
        gap |= not active
        uncertain |= any(r.get('uncertain') for r in active)
        if len(current) > 1 or any(r.get('overlap') for r in active):
            overlap.extend(active)
    if overlap:
        return _overlap(overlap), 'acoustic_overlap'
    if len(labels) > 1:
        return '归属待核对', 'crosses_voice_boundary'
    if gap or not labels:
        return '归属待核对', 'incomplete_acoustic_coverage'
    if uncertain:
        return '归属待核对', 'uncertain_voice'
    return next(iter(labels)), 'native_interval'


def align_dialogue_segments(segments, tokens, acoustic_turns, *, prefix):
    output, total_matched, total_chars = [], 0, 0
    for source in segments:
        original = source['text']
        candidates = []
        for token in tokens:
            start, end = token.get('start_ms'), token.get('end_ms')
            if (not isinstance(start, (int, float)) or not math.isfinite(start) or
                    not source['start_ms'] <= start < source['end_ms']):
                continue
            if end is not None and (not isinstance(end, (int, float)) or not math.isfinite(end) or
                                    end < start or end > source['end_ms']):
                continue
            # Missing end is a legitimate onset-only token. A supplied
            # zero-width end is failed duration evidence and stays zero.
            candidates.append({**token, 'end_ms': end})
        candidates.sort(key=lambda t: t['start_ms'])
        reference, times = [], []
        for index, token in enumerate(candidates):
            review_end = token['end_ms']
            if review_end is None:
                review_end = next((t['start_ms'] for t in candidates[index + 1:]
                                   if t['start_ms'] > token['start_ms']), source['end_ms'])
            for char in token['text']:
                if not char.isspace():
                    reference.append(char.casefold())
                    times.append((token['start_ms'], token['end_ms'], review_end))
        positions = [i for i, char in enumerate(original) if not char.isspace()]
        matcher = difflib.SequenceMatcher(None, [original[i].casefold() for i in positions], reference, autojunk=False)
        anchors = {}
        for a, b, size in matcher.get_matching_blocks():
            for offset in range(size):
                anchors[positions[a + offset]] = times[b + offset]
        sorted_anchors = sorted(anchors)
        mapped = []
        for index, char in enumerate(original):
            native_start = native_end = None
            if index in anchors:
                start, native_end, end = anchors[index]
                native_start = start
                speaker, reason = (_voice_at(start, acoustic_turns) if native_end is None else
                                   _voice_between(start, native_end, acoustic_turns))
            else:
                previous = next((i for i in reversed(sorted_anchors) if i < index), None)
                following = next((i for i in sorted_anchors if i > index), None)
                start = ((anchors[previous][1] if anchors[previous][1] is not None else anchors[previous][0])
                         if previous is not None else source['start_ms'])
                end = anchors[following][0] if following is not None else source['end_ms']
                start, end = min(start, end), max(start, end)
                speaker, reason = _voice_between(start, end, acoustic_turns)
                if speaker.startswith('交叠'):
                    speaker, reason = '归属待核对', 'unaligned_overlap'
            if (char.isspace() or unicodedata.category(char)[0] == 'P') and mapped:
                speaker, reason = mapped[-1]['speaker'], 'punctuation_context'
            evidence = {'segment_id': str(source.get('id', '')),
                        'source_text_sha256': hashlib.sha256(original.encode('utf-8')).hexdigest(),
                        'text_start': index, 'text_end': index + 1,
                        'native_start_ms': native_start, 'native_end_ms': native_end,
                        'review_start_ms': start, 'review_end_ms': end, 'assignment_reason': reason}
            mapped.append({'text': char, 'speaker': speaker, 'start_ms': start, 'end_ms': end,
                           'evidence': evidence, 'anchored': index in anchors})
        total_matched += len(anchors)
        total_chars += len(original)
        for item in mapped:
            if output and output[-1]['speaker'] == item['speaker'] and item['start_ms'] - output[-1]['end_ms'] < 800:
                row = output[-1]
                row['text'] += item['text']
                row['start_ms'] = min(row['start_ms'], item['start_ms'])
                row['end_ms'] = max(row['end_ms'], item['end_ms'])
                row['native_anchor_chars'] += item['anchored']
                row['alignment_spans'].append(item['evidence'])
                if str(source.get('id', '')) not in row['segment_ids']:
                    row['segment_ids'].append(str(source.get('id', '')))
            else:
                output.append({'id': f'{prefix}-ctc-{len(output) + 1:03}', 'text': item['text'],
                    'speaker': item['speaker'], 'start_ms': item['start_ms'], 'end_ms': item['end_ms'],
                    'segment_ids': [str(source.get('id', ''))], 'alignment_spans': [item['evidence']],
                    'native_anchor_chars': int(item['anchored']), 'timing_precision': 'review-envelope',
                    'annotation_kind': 'text-review-envelope'})
    for row in output:
        # Numeric legacy offsets are playback envelopes, not word durations.
        row['review_start_ms'], row['review_end_ms'] = row['start_ms'], row['end_ms']
    assert ''.join(r['text'] for r in output) == ''.join(r['text'] for r in segments)
    return output, {'source_chars': total_chars, 'native_anchor_chars': total_matched,
                    'anchor_ratio': total_matched / total_chars if total_chars else None,
                    'word_timestamps_generated': False, 'offsets_kind': 'review-envelope',
                    'native_onset_policy': 'half-open-point', 'scoring_eligible': False}
