"""Run private, same-input polish comparisons; stdout contains only a report path.

Use an installed NextEcho runtime. Cases are JSON {"cases": [{"id", "text",
"suite"}]} and stay outside the distribution. No datasets or models download.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from workbench import postprocess as P
from workbench.config import NextEchoConfig, load_config
from workbench.enhancement import default_glossary_rules
from workbench.io_utils import file_hash, value_hash, write_json
from workbench.model_profiles import PROFILES
from workbench.owned_provider import owned_provider
from workbench.resources import private_dir


MODELS = ("qwen35-2b", "qwen35-4b", "qwen35-9b")


def evaluate_case(case, target, config, profile):
    source = P.Segment(str(case["id"]), case["text"], 0, 30000, "speaker_1")
    write_json(target / "transcript.json", {
        "transcription": [P.segment_to_transcription(source)],
        "coverage": {"covered_seconds": 30, "partial": False},
    })
    original = file_hash(target / "transcript.json")
    attempts = []
    real_call = P.call_json

    def capture(**kwargs):
        attempt = {"raw_guard_pass": False}
        try:
            result = real_call(**kwargs)
            attempt["response"] = result[0]
            try:
                P.validate_polish_response([source], result[0], default_glossary_rules())
                attempt["raw_guard_pass"] = True
            except Exception as exc:
                attempt["validation_error"] = str(exc)
            return result
        except Exception as exc:
            attempt["error_code"] = getattr(exc, "code", type(exc).__name__)
            raise
        finally:
            attempts.append(attempt)
            write_json(target / "raw_attempts.json", attempts)

    with patch.object(P, "call_json", side_effect=capture):
        audit = P.process_transcript_artifacts(target, profile=profile, config=config, summarize=False)
    output = P.load_segments(json.loads((target / "transcript.enhanced.json").read_text()), layer="enhanced")
    protected = len(output) == 1
    if protected:
        row = output[0]
        protected = (row.id, row.start_ms, row.end_ms, row.speaker) == (
            source.id, source.start_ms, source.end_ms, source.speaker)
        try:
            P.ensure_preserved(source.text, row.text, default_glossary_rules())
        except Exception:
            protected = False
    result = {
        "id": case["id"], "suite": case.get("suite", "custom"),
        "final_protected": protected,
        "protected_fields_pass": protected,
        "original_unchanged": original == file_hash(target / "transcript.json"),
        "raw_first_pass": bool(attempts and attempts[0]["raw_guard_pass"]),
        "required_fallback": bool(audit.get("polish", {}).get("original_fallback_chunks", [])),
        "review_required": bool(audit.get("polish", {}).get("review_required", False)),
        "attempts": len(attempts),
        "changed": len(output) == 1 and output[0].text != source.text,
        "output_text": output[0].text if len(output) == 1 else None,
        "audit": str(target / "postprocess.audit.json"), "human_review": "pending",
    }
    write_json(target / "evaluation.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="受信任的本机安装配置，包含服务及资源预算")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=range(1, 11), default=10)
    args = parser.parse_args(argv)
    cases = json.loads(args.cases.read_text())["cases"]
    if not cases or any(not isinstance(c.get("id"), str) or not c["id"] or
                        not isinstance(c.get("text"), str) or not c["text"].strip() for c in cases):
        parser.error("cases必须包含非空字符串id和text")
    if len({c["id"] for c in cases}) != len(cases):
        parser.error("case id必须唯一")
    target = args.output_dir.expanduser().resolve()
    if target.exists() and any(target.iterdir()):
        parser.error("使用新的空输出目录，避免混入旧模型/设置的结果")
    private_dir(target)
    config = load_config(args.config, strict=True)
    report = {"schema_version": 1, "selection_sha256": value_hash(cases),
              "cases_sha256": file_hash(args.cases), "config_sha256": file_hash(args.config),
              "batch_size": args.batch_size, "postprocess_sha256": file_hash(Path(P.__file__)),
              "models": {}, "human_review": "pending", "reviewers_required": 1,
              "release_status": "candidate", "deferred_models": ["27B", "35B MoE"]}
    failed = False
    for name in dict.fromkeys(args.models):
        model = PROFILES[name]
        service = config.get_path("providers", "ollama", default={})
        endpoint = service.get("endpoint", "http://127.0.0.1:11434")
        settings = config.to_dict()
        settings["postprocess"].update(provider="ollama", endpoint=endpoint,
            preferred_model_id=model.model, fallback_model=model.model, quality_model=model.model,
            strict_selection=True, ownership="external")
        lane = report["models"][name] = {"status": "running", "cases": []}
        try:
            for start in range(0, len(cases), args.batch_size):
                with owned_provider(service, endpoint=endpoint, model=model.model, config=config):
                    for index in range(start, min(start + args.batch_size, len(cases))):
                        directory = private_dir(target / name / f"{index + 1:05}")
                        lane["cases"].append(evaluate_case(cases[index], directory, NextEchoConfig(settings), model.tier))
                        write_json(target / "summary.json", report)
            lane["status"] = "evaluated"
        except Exception as exc:
            lane.update(status="incomplete", error_code=getattr(exc, "code", type(exc).__name__))
            failed = True
        for suite in sorted({c.get("suite", "custom") for c in cases}):
            rows = [r for r in lane["cases"] if r["suite"] == suite]
            lane.setdefault("metrics", {})[suite] = {
                "n": len(rows), "final_protected": sum(r["final_protected"] for r in rows),
                "protected_fields_pass": sum(r["protected_fields_pass"] for r in rows),
                "review_required_n": sum(r["review_required"] for r in rows),
                "original_unchanged": sum(r["original_unchanged"] for r in rows),
                "raw_first_pass": sum(r["raw_first_pass"] for r in rows),
                "fallback_n": sum(r["required_fallback"] for r in rows),
                "no_fallback_rate": sum(not r["required_fallback"] for r in rows) / len(rows) if rows else None,
            }
        write_json(target / "summary.json", report)
    print(target / "summary.json")
    return 5 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
