"""Deterministic high-risk edit checks; never a semantic correctness proof."""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from decimal import Decimal, InvalidOperation, localcontext
from difflib import SequenceMatcher

from .enhancement import strip_filler_words
from .chinese_text import simplified

VALIDATOR_VERSION = "protected-fields-v2"
_DIGITS = {c: n for n, chars in enumerate(("零〇", "一", "二两", "三", "四", "五", "六", "七", "八", "九")) for c in chars}
_SMALL = {"十": 10, "百": 100, "千": 1000}
_BIG = {"亿": 100000000, "万": 10000}
_NUMBER = r"(?:\d+(?:\.\d+)?(?:万亿|[万亿])?|[零〇一二两三四五六七八九十百千万亿点]+)"
_UNITS = {
    "亿元": ("元", 100000000), "万元": ("元", 10000), "元": ("元", 1),
    "人民币": ("人民币", 1), "美元": ("美元", 1),
    **{s: (s, 1) for s in ("GiB", "MiB", "GB", "MB", "公斤", "千克", "厘米", "毫升", "分钟", "小时", "秒", "米", "年", "月", "日", "人", "个", "次", "倍", "万", "亿", "%", "％")},
}
_QUANTITY = re.compile(
    r"(?P<prefix>大约|大概|将近|超过|至少|至多|不到|大于|小于|约|近)?"
    r"(?P<percent>百分之)?(?P<sign>负|[-＋－+])?(?P<number>" + _NUMBER + r")\s*"
    r"(?P<unit>" + "|".join(sorted(_UNITS, key=len, reverse=True)) + r")?"
    r"(?P<suffix>左右|上下|多|余)?"
)
_LITERAL = re.compile(r"```[\s\S]*?```|`[^`\n]*`|https?://[^\s<>]+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
_CRITICAL = re.compile("|".join(sorted({
    "不能", "不会", "不应", "不是", "不支持", "不允许", "不建议", "不赞成", "没有", "未完成", "未通过",
    "禁止", "不得", "除非", "仅当", "如果", "只有", "只要", "否则", "但是", "但", "不", "未", "没", "无",
    "可能", "也许", "或许", "大概", "必然", "一定", "肯定", "必定", "大概率", "小概率",
    "亏损", "盈利", "利润", "收入", "支出", "成本", "资产", "负债", "现金", "同比", "环比",
    "增加", "减少", "增长", "下降", "上涨", "降低", "上升", "高于", "低于",
    "赞成", "建议", "支持", "反对", "通过", "同意", "拒绝", "允许", "必须", "应该",
}, key=lambda s: (-len(s), s))))


def _integer(raw: str, *, serial: bool = False) -> int:
    for unit, factor in _BIG.items():
        if unit in raw:
            if raw.count(unit) != 1:
                raise ValueError("ambiguous repeated scale")
            left, right = raw.split(unit)
            if right and right[0] in _DIGITS and right[0] not in "零〇" and all(c in _DIGITS for c in right):
                raise ValueError("ambiguous elided trailing place")
            right = right.lstrip("零〇")
            return _integer(left or "一") * factor + (_integer(right) if right else 0)
    if all(c in _DIGITS for c in raw):
        if len(raw) > 1 and not serial:
            raise ValueError("unqualified adjacent digits")
        return int("".join(str(_DIGITS[c]) for c in raw))
    total, digit, previous, last_unit = 0, 0, 10000, -1
    for index, c in enumerate(raw):
        if c in _DIGITS:
            digit = _DIGITS[c]
        elif c in _SMALL and _SMALL[c] < previous:
            previous = _SMALL[c]
            last_unit = index
            total += (digit or 1) * previous
            digit = 0
        else:
            raise ValueError("unsupported number form")
    if digit and previous > 10 and not any(c in "零〇" for c in raw[last_unit + 1:]):
        raise ValueError("ambiguous elided trailing place")
    return total + digit


def _decimal(raw: str, *, serial: bool = False) -> Decimal:
    if len(raw) > 64:
        raise ValueError("number form exceeds certified grammar")
    ascii_number = re.fullmatch(r"(\d+(?:\.\d+)?)(万亿|[万亿])?", raw)
    if ascii_number:
        return Decimal(ascii_number[1]) * {None: 1, "万": 10000, "亿": 100000000, "万亿": 1000000000000}[ascii_number[2]]
    if "点" in raw:
        for suffix, factor in (("万亿", 1000000000000), ("亿", 100000000), ("万", 10000)):
            if raw.endswith(suffix):
                return _decimal(raw[:-len(suffix)]) * factor
        left, right = raw.split("点")
        if not right or any(c not in _DIGITS for c in right):
            raise ValueError("unsupported decimal")
        return Decimal(str(_integer(left or "零")) + "." + "".join(str(_DIGITS[c]) for c in right))
    return Decimal(_integer(raw, serial=serial))


def _format(value: Decimal) -> str:
    return format(value, "f").rstrip("0").rstrip(".") if "." in format(value, "f") else str(value)


def canonical_quantities(text: str) -> tuple[str, list[tuple]]:
    """Normalize only supported numeric forms, retaining approximation/unit.

    Unknown/ambiguous forms keep a literal signature. This bounded grammar
    does not certify every Chinese numeral or allow arbitrary paraphrases.
    URL and code literals are not rewritten.
    """
    # Comparison form only: original/adopted artifacts are never rewritten here.
    text = simplified(text)
    signatures: list[tuple] = []

    def replace(match):
        raw, unit = match["number"], match["unit"] or ""
        normalized_unit, factor = _UNITS.get(unit, (unit, 1))
        if unit in {"万", "亿"}:
            normalized_unit, factor = "", _BIG[unit]
        if match["percent"]:
            if unit:
                return match[0]  # unsupported mixed percent/unit, preserve it
            normalized_unit = "%"
        if normalized_unit == "％":
            normalized_unit = "%"
        prefix, suffix = match["prefix"] or "", match["suffix"] or ""
        sign = -1 if match["sign"] in {"负", "-", "－"} else 1
        literal_context = bool(re.search(r"(?:版本|型号|编号|序号|第)\s*$", match.string[max(0, match.start() - 6):match.start()]))
        if not (unit or match["percent"] or prefix or suffix or raw.endswith(("万", "亿")) or literal_context) and not raw[0].isascii():
            # An unqualified numeral embedded in Chinese prose can be lexical
            # (一度/一直/不一样), not a certified numeric field.
            return match[0]
        try:
            with localcontext() as decimal_context:
                decimal_context.prec = 160
                if (not (unit or match["percent"] or prefix or suffix or raw.endswith(("万", "亿")))
                        or literal_context):
                    raise ValueError("unqualified numeral or literal identifier/version")
                adjacent = re.fullmatch(r"([一二两三四五六七八九])([一二两三四五六七八九])([十百千万亿]*)", raw)
                if adjacent and _DIGITS[adjacent[2]] == _DIGITS[adjacent[1]] + 1 and (adjacent[3] or factor > 1):
                    scale = _decimal(adjacent[3]) if adjacent[3] else Decimal(1)
                    values = tuple(_format(Decimal(_DIGITS[adjacent[n]]) * scale * factor * sign) for n in (1, 2))
                    kind = "approximate-range"
                else:
                    values = (_format(_decimal(raw, serial=unit == "年") * factor * sign),)
                    kind = "scalar"
        except (ValueError, InvalidOperation):
            kind, values = "literal-number", (raw, match["sign"] or "")
        signature = (kind, values, normalized_unit, prefix, suffix)
        signatures.append(signature)
        if kind == "literal-number":
            return match[0]
        rendered = values[0] if kind == "scalar" else "约数" + "至".join(values)
        return prefix + rendered + normalized_unit + suffix

    parts, cursor = [], 0
    for literal in _LITERAL.finditer(text):
        parts.extend((_QUANTITY.sub(replace, text[cursor:literal.start()]), literal[0]))
        cursor = literal.end()
    parts.append(_QUANTITY.sub(replace, text[cursor:]))
    return "".join(parts), signatures


def high_risk_edits(before: str, after: str) -> list[str]:
    before, quantities_before = canonical_quantities(before)
    after, quantities_after = canonical_quantities(after)
    risks = []
    if _LITERAL.findall(before) != _LITERAL.findall(after):
        risks.append("PROTECTED_LITERAL_CHANGED")
    if quantities_before != quantities_after:
        risks.append("QUANTITY_VALUE_UNIT_OR_APPROXIMATION_CHANGED")
    def english(value):
        # Numbers have their own check. Sentence periods and spacing before
        # dates must not turn OK2023/OK, 2023 or Social/Social. into new words.
        return [token.casefold() for token in re.findall(r"[A-Za-z]+(?:[._+-]+[A-Za-z]+)*(?:\+\+|#)?", value)]
    if english(before) != english(after):
        risks.append("ENGLISH_TOKEN_CHANGED")
    markers_before, markers_after = list(_CRITICAL.finditer(before)), list(_CRITICAL.finditer(after))
    if [m[0] for m in markers_before] != [m[0] for m in markers_after]:
        risks.append("NEGATION_MODALITY_CONDITION_OR_FINANCIAL_RELATION_CHANGED")
    else:
        def scopes(value):
            # Ignore punctuation and known mechanical startup repetitions;
            # preserve nearby subject/predicate anchors conservatively.
            value = strip_filler_words(value)
            for source, target in (("我我们", "我们"), ("你你们", "你们"), ("我我", "我"), ("你你", "你"),
                                   ("他他", "他"), ("她她", "她"), ("这个这个", "这个"), ("那个那个", "那个")):
                value = value.replace(source, target)
            compact = re.sub(r"[^0-9A-Za-z\u3400-\u9fff]+", "", value).casefold()
            return [(m[0], compact[max(0, m.start() - 8):m.start()], compact[m.end():m.end() + 12]) for m in _CRITICAL.finditer(compact)]
        if scopes(before) != scopes(after):
            risks.append("HIGH_RISK_SCOPE_OR_SUBJECT_EDIT")
    return risks


def blocking_risk_edits(before: str, after: str) -> list[str]:
    """Separate explicit protected-field changes from uncertain scope edits.

    Scope edits still require human review. A word/argument permutation with
    the same character inventory is blocked rather than silently reassigned.
    """
    risks = high_risk_edits(before, after)
    blocking = [risk for risk in risks if risk != "HIGH_RISK_SCOPE_OR_SUBJECT_EDIT"]
    if "HIGH_RISK_SCOPE_OR_SUBJECT_EDIT" in risks:
        left, _ = canonical_quantities(before)
        right, _ = canonical_quantities(after)
        def compact(text):
            return re.sub(r"[^0-9A-Za-z\u3400-\u9fff]+", "", text).casefold()
        if Counter(compact(left)) == Counter(compact(right)):
            blocking.append("HIGH_RISK_SCOPE_OR_SUBJECT_EDIT")
    return blocking


def text_patch_evidence(before: str, after: str, *, segment_id: str, layer: str) -> dict:
    """Unicode character offsets and exact hashes for an inspectable edit."""
    return {
        "segment_id": segment_id, "layer": layer, "validator_version": VALIDATOR_VERSION,
        "source_sha256": hashlib.sha256(before.encode()).hexdigest(),
        "target_sha256": hashlib.sha256(after.encode()).hexdigest(),
        "risk_tags": high_risk_edits(before, after), "semantic_review": "pending",
        "edits": [{"operation": tag, "source_start": start, "source_end": end, "target_start": target_start, "target_end": target_end,
                   "removed": before[start:end], "replacement": after[target_start:target_end]}
                  for tag, start, end, target_start, target_end in SequenceMatcher(None, before, after, autojunk=False).get_opcodes() if tag != "equal"],
    }
