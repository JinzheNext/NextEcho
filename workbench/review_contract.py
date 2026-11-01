"""Content-bound audio and human annotations; no model-quality inference."""
from __future__ import annotations

from datetime import datetime
import math
import os
from pathlib import Path
import re
from urllib.parse import quote, unquote, urlsplit
import wave

from .io_utils import file_hash, value_hash

SCHEMA = 3
FIELDS = {
    "identity": {"verdict": {"pending", "pass", "wrong", "uncertain"}, "note": None},
    "polish": {"meaning": {"pending", "pass", "fail", "uncertain"},
               "readability": {"pending", "better", "same", "worse"}, "note": None,
               "voice_flag": {"pending", "flagged"}, "voice_note": None},
}


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def inspect_audio(path: Path, *, expected_sha256: str, origin_ms: int) -> dict:
    if type(origin_ms) is not int or origin_ms < 0:
        raise ValueError("审核音频必须声明非负整数origin_ms")
    if not isinstance(expected_sha256, str) or not re.fullmatch("[0-9a-f]{64}", expected_sha256):
        raise ValueError("审核音频需要独立的预期SHA-256，不能仅记录当前文件hash")
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = file_hash(path)
    if actual != expected_sha256:
        raise ValueError("审核音频与预期hash不符")
    try:
        with wave.open(str(path), "rb") as f:
            frames, rate, channels, width = f.getnframes(), f.getframerate(), f.getnchannels(), f.getsampwidth()
            if not frames or not rate or f.getcomptype() != "NONE":
                raise ValueError("审核音频必须是非空PCM WAV")
            count = 0
            while raw := f.readframes(65536):
                count += len(raw)
            if count != frames * channels * width:
                raise ValueError("审核音频PCM被截断")
    except (wave.Error, EOFError) as exc:
        raise ValueError("审核音频不是有效PCM WAV") from exc
    # Catch mutation between identity and waveform inspection; no read of text.
    if file_hash(path) != actual:
        raise ValueError("审核音频在校验中改变")
    return {"sha256": actual, "origin_ms": origin_ms, "frames": frames,
            "sample_rate": rate, "channels": channels, "sample_width": width,
            "duration_ms": frames / rate * 1000, "verification": "build-time-hash-and-pcm"}


def interval(media: dict, start_ms, end_ms, *, allow_point: bool = False) -> tuple[float, float]:
    origin = media["origin_ms"]
    if (not number(start_ms) or not number(end_ms) or start_ms < origin
            or (start_ms > end_ms if allow_point else start_ms >= end_ms)
            or end_ms > origin + media["duration_ms"] + 1e-6):
        raise ValueError("审核区间超出音频真实时长或origin范围")
    return (start_ms - origin) / 1000, (end_ms - origin) / 1000


def local_url(path: Path, page: Path) -> str:
    return quote(os.path.relpath(path.resolve(), page.parent.resolve()), safe="/")


def record_media(record: dict, page: Path, cache: dict) -> tuple[str, dict | None]:
    reference = record.get("audio")
    if not reference:
        if record.get("media"):
            raise ValueError("审核媒体声明缺少本地音频路径")
        return "", None
    if not isinstance(reference, str):
        raise ValueError("审核音频路径无效")
    if reference.startswith("/") and not reference.startswith("//"):
        # An explicit absolute filesystem path can contain literal # and ?.
        path = Path(reference)
    else:
        parsed = urlsplit(reference)
        if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment or "\\" in reference:
            raise ValueError("审核音频只接受本地文件，不能执行网络或脚本URL")
        path = Path(unquote(parsed.path))
    if not path.is_absolute():
        path = page.parent / path
    declared = record.get("media")
    if not isinstance(declared, dict):
        raise ValueError("带音频的审核条目必须声明media.sha256/origin_ms/start_ms/end_ms")
    if not isinstance(declared.get("sha256"), str) or type(declared.get("origin_ms")) is not int:
        raise ValueError("审核media哈希/origin类型不符")
    key = (str(path.resolve()), declared.get("sha256"), declared.get("origin_ms"))
    if key not in cache:
        cache[key] = inspect_audio(path, expected_sha256=declared.get("sha256"), origin_ms=declared.get("origin_ms"))
    media = {**cache[key], "start_ms": declared.get("start_ms"), "end_ms": declared.get("end_ms")}
    start, end = interval(media, media["start_ms"], media["end_ms"])
    for name, expected in (("audio_start", start), ("audio_end", end)):
        if name in record and (not number(record[name]) or not math.isclose(record[name], expected, abs_tol=1e-6)):
            raise ValueError("旧试听秒数与显式media全局时间/origin不符")
    return local_url(path, page), media


def make_item(kind: str, source: dict, content) -> dict:
    return {"item_id": value_hash([SCHEMA, kind, source, content]), "source": source}


def model_reference(value) -> dict:
    allowed = {"status", "model", "repo", "revision", "sha256", "digest", "runtime", "quantization", "adopted_file_sha256"}
    reference = {"status": "unknown"}
    if isinstance(value, dict):
        reference.update({k: v for k, v in value.items() if k in allowed and isinstance(v, str) and len(v) <= 256})
    return reference


def make_contract(kind: str, items: list[dict]) -> dict:
    if kind not in FIELDS or len({x["item_id"] for x in items}) != len(items):
        raise ValueError("审核类型或条目版本重复")
    return {"schema_version": SCHEMA, "review_kind": kind, "review_items": items,
            "fingerprint": value_hash([SCHEMA, kind, sorted(items, key=lambda x: x["item_id"])])}


def validate_annotations(export: dict, dataset: dict) -> dict:
    """Validate provenance/schema only, never declares human opinions correct."""
    if (not isinstance(export, dict) or not isinstance(dataset, dict)
            or dataset.get("schema_version") != SCHEMA or dataset.get("review_kind") not in FIELDS):
        raise ValueError("审核数据集/导出不是当前版本对象")
    fields = FIELDS[dataset["review_kind"]]
    required = {"schema_version", "artifact_kind", "review_kind", "dataset_fingerprint", "reviewer", "reviewed_at", "items"}
    if (set(export) != required or export["schema_version"] != SCHEMA
            or export["artifact_kind"] != "nextecho-human-review"
            or export["review_kind"] != dataset["review_kind"]
            or export["dataset_fingerprint"] != dataset["fingerprint"]
            or not isinstance(export["reviewer"], str) or not 0 < len(export["reviewer"].strip()) <= 80
            or not isinstance(export["reviewed_at"], str) or len(export["reviewed_at"]) > 40
            or not isinstance(export["items"], list)):
        raise ValueError("审核导出类型、版本、审核人或数据集不符")
    if datetime.fromisoformat(export["reviewed_at"].replace("Z", "+00:00")).tzinfo is None:
        raise ValueError("审核时间必须含时区")
    expected = {item["item_id"]: item["source"] for item in dataset["review_items"]}
    seen = set()
    for row in export["items"]:
        if not isinstance(row, dict) or set(row) != {"item_id", "source", "decision"}:
            raise ValueError("审核导出含非白名单条目")
        key = row["item_id"]
        if not isinstance(key, str) or key in seen or key not in expected or row["source"] != expected[key]:
            raise ValueError("审核导出包含旧条目、重复条目或来源改变")
        seen.add(key)
        decision = row["decision"]
        if not isinstance(decision, dict) or set(decision) != set(fields):
            raise ValueError("审核判断字段不符")
        for name, allowed in fields.items():
            value = decision[name]
            if not isinstance(value, str) or (allowed is not None and value not in allowed) or (allowed is None and len(value) > 4000):
                raise ValueError("审核判断或批注无效")
    if seen != set(expected):
        raise ValueError("审核导出缺少当前条目，未审核也须显式pending")
    return export


# Pure functions are also exercised in Node; DOM/storage/playback remain the
# pages' responsibility. No arbitrary localStorage object is exported.
ANNOTATION_JS = r'''
const ReviewAnnotations=(()=>{
 const fields={identity:{verdict:['pending','pass','wrong','uncertain'],note:null},polish:{meaning:['pending','pass','fail','uncertain'],readability:['pending','better','same','worse'],note:null,voice_flag:['pending','flagged'],voice_note:null}};
 function restore(raw,kind){try{const v=JSON.parse(raw||'null');if(v&&v.schema_version===3&&v.review_kind===kind&&v.answers&&typeof v.answers==='object'&&!Array.isArray(v.answers))return Object.assign(Object.create(null),v.answers)}catch(e){}return Object.create(null)}
 function state(answers,kind){return {schema_version:3,review_kind:kind,answers}}
 function clean(item,saved,kind){const d={};for(const [field,allowed] of Object.entries(fields[kind])){const v=saved[item.item_id+':'+field];d[field]=typeof v==='string'&&(allowed?allowed.includes(v):v.length<=4000)?v:allowed?'pending':''}return d}
 function exportReview(data,saved,reviewer,now){reviewer=reviewer.trim();if(!reviewer||reviewer.length>80)throw Error('请填写1–80字审核人或代号');return {schema_version:3,artifact_kind:'nextecho-human-review',review_kind:data.review_kind,dataset_fingerprint:data.fingerprint,reviewer,reviewed_at:now,items:data.review_items.map(item=>({item_id:item.item_id,source:item.source,decision:clean(item,saved,data.review_kind)}))}}
 return {restore,state,clean,exportReview};
})();
'''
