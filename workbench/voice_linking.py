"""Conservative acoustic-only cross-window linking, with no speaker-count oracle."""
from __future__ import annotations

import math


def link_voice_prototypes(nodes: list[dict], *, threshold: float = .8) -> tuple[dict, dict]:
    if not .5 <= threshold <= 1:
        raise ValueError("global voice cosine threshold must be between 0.5 and 1")
    keys = [r["key"] for r in nodes]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate voice prototype keys")
    vectors = []
    for row in nodes:
        vector = row.get("embedding") or []
        norm = math.sqrt(sum(float(v)**2 for v in vector))
        vectors.append([float(v)/norm for v in vector] if norm > 0 and math.isfinite(norm) else [])
    similarities = {}
    for a in range(len(nodes)):
        for b in range(a+1, len(nodes)):
            va, vb = vectors[a], vectors[b]
            similarities[a, b] = sum(x*y for x, y in zip(va, vb)) if va and len(va) == len(vb) else -1
    groups = [[i] for i in range(len(nodes))]
    def score(left, right):
        if {nodes[i]["chunk"] for i in left} & {nodes[i]["chunk"] for i in right}:
            return -1  # Two distinct local clusters cannot silently become one.
        return min(similarities[min(a,b), max(a,b)] for a in left for b in right)
    while True:
        best = None
        for a in range(len(groups)):
            for b in range(a+1, len(groups)):
                value = score(groups[a], groups[b])
                if value >= threshold and (best is None or value > best[0]):
                    best = value, a, b
        if best is None:
            break
        _, a, b = best
        groups[a] += groups.pop(b)
    groups.sort(key=lambda group: min((nodes[i]["first"], keys[i]) for i in group))
    mapping, records = {}, []
    for index, group in enumerate(groups):
        label = f"GLOBAL_{index:03d}"
        for i in group:
            mapping[keys[i]] = label
        pairs = [similarities[min(a,b), max(a,b)] for a in group for b in group if a < b]
        records.append({"label": label, "members": [keys[i] for i in sorted(group)],
                        "minimum_pair_cosine": min(pairs) if pairs else None,
                        "insufficient_embedding": any(not vectors[i] for i in group)})
    return mapping, {"method": "constrained-complete-link-cosine", "threshold": threshold,
                     "score_kind": "acoustic_cosine_not_identity_probability", "groups": records,
                     "local_prototypes": len(nodes), "global_clusters": len(groups)}


def link_acoustic_measurements(measurements: list[dict], *, threshold: float = .8) -> dict:
    """Average-link prototypes; only simultaneous audio supplies cannot-links.

    Distinct local cluster numbers are model predictions, not proof of distinct
    people. Keeping all of them mutually exclusive made local over-splitting
    irreversible. Actual overlapping intervals remain explicit constraints.
    """
    if not .5 <= threshold <= 1:
        raise ValueError("global voice cosine threshold must be between 0.5 and 1")
    nodes = [p for m in measurements for p in m["prototypes"]]
    keys = [n["key"] for n in nodes]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate measured voice keys")
    vectors = []
    for n in nodes:
        vector = n.get("embedding") or []
        norm = math.sqrt(sum(float(v)**2 for v in vector))
        vectors.append([float(v)/norm for v in vector] if norm > 0 and math.isfinite(norm) else [])
    indices = {key:i for i,key in enumerate(keys)}
    cannot = set()
    for m in measurements:
        turns = m["turns"]
        for a, left in enumerate(turns):
            for right in turns[a+1:]:
                if left["speaker"] == right["speaker"]:
                    continue
                if min(left["end"], right["end"])-max(left["start"], right["start"]) >= .5:
                    i = indices[f"{m['index']:04d}:{left['speaker']}"]
                    j = indices[f"{m['index']:04d}:{right['speaker']}"]
                    cannot.add((min(i,j), max(i,j)))
    similarities = {}
    for a in range(len(nodes)):
        for b in range(a+1, len(nodes)):
            va, vb = vectors[a], vectors[b]
            similarities[a,b] = sum(x*y for x,y in zip(va,vb)) if va and len(va) == len(vb) else -1
    groups = [[i] for i in range(len(nodes))]
    while True:
        best = None
        for a in range(len(groups)):
            for b in range(a+1,len(groups)):
                pairs = [(min(i,j),max(i,j)) for i in groups[a] for j in groups[b]]
                if any(p in cannot for p in pairs):
                    continue
                score = sum(similarities[p] for p in pairs)/len(pairs)
                if score >= threshold and (best is None or score > best[0]):
                    best = score,a,b
        if best is None:
            break
        _,a,b = best
        groups[a] += groups.pop(b)
    groups.sort(key=lambda group:min((nodes[i]["first"],keys[i]) for i in group))
    mapping, records = {},[]
    for index, group in enumerate(groups):
        label = f"VOICE_{index:03d}"
        for i in group:
            mapping[keys[i]] = label
        pairs = [similarities[min(i,j),max(i,j)] for i in group for j in group if i < j]
        records.append({"label":label,"members":[keys[i] for i in sorted(group)],
                        "mean_pair_cosine":sum(pairs)/len(pairs) if pairs else None,
                        "minimum_pair_cosine":min(pairs) if pairs else None,
                        "insufficient_embedding":any(not vectors[i] for i in group)})
    output = []
    for m in measurements:
        for t in m["turns"]:
            key = f"{m['index']:04d}:{t['speaker']}"
            label = mapping[key]
            output.append({**t,"speaker":label,"raw_speaker_label":label,"source_acoustic_key":key})
    output.sort(key=lambda t:(t["start"],t["end"],t["speaker"]))
    return {"turns":output,"global_linking":{"method":"overlap-constrained-average-link-cosine",
            "threshold":threshold,"cannot_link_minimum_overlap_seconds":.5,"cannot_link_pairs":len(cannot),
            "score_kind":"acoustic_cosine_not_identity_probability","groups":records,
            "local_prototypes":len(nodes),"global_clusters":len(groups)}}
