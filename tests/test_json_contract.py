from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from workbench.llm_providers import CONTEXT, GenerationContext, decode_model_json, validate_json_schema
from workbench.postprocess import call_json, polish_schema
from workbench.config import NextEchoConfig


class JsonContractTests(unittest.TestCase):
    def test_duplicate_keys_and_overflow_are_not_silently_adopted(self):
        for text in ('{"segments":[],"segments":[{}]}', '{"s":{"id":"a","id":"b"}}',
                     r'{"id":"a","\u0069d":"b"}', '{"score":1e309}', '{"score":-1e999}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                decode_model_json(text)
        self.assertEqual(decode_model_json('{"first":{"id":"a"},"second":{"id":"b"}}')[0],
                         {"first": {"id": "a"}, "second": {"id": "b"}})

    def test_closed_polish_objects_reject_unknown_root_and_segment_fields(self):
        row = {"id": "s", "polished_text": "原文保留。", "uncertain_terms": [], "change_notes": []}
        for contextual in (False, True):
            token = CONTEXT.set(GenerationContext({"contextual_polish": contextual}))
            try:
                current = {key: value for key, value in row.items() if not contextual or key in {"id", "polished_text"}}
                validate_json_schema({"segments": [current]}, polish_schema())
                for value in ({"segments": [current], "replacement": "unexpected"},
                              {"segments": [{**current, "replacement": "unexpected"}]}):
                    with self.subTest(contextual=contextual, value=value), self.assertRaises(ValueError):
                        validate_json_schema(value, polish_schema())
            finally:
                CONTEXT.reset(token)

    def test_additional_properties_schema_is_applied_recursively(self):
        schema = {"type": "object", "additionalProperties": {"type": "array", "items": {"type": "string"}}}
        validate_json_schema({"names": ["speaker"]}, schema)
        with self.assertRaises(ValueError):
            validate_json_schema({"names": [1]}, schema)

    def test_both_transports_validate_even_without_strict_model_selection(self):
        schema = {"type": "object", "required": ["value"], "properties": {"value": {"type": "string"}},
                  "additionalProperties": False}
        for provider in ("ollama", "local-chat-compatible"):
            for mode in ("native-schema", "prompt-json-validated"):
                for strict in (False, True):
                    token = CONTEXT.set(GenerationContext({"provider": provider, "structured_mode": mode,
                        "strict_selection": strict}))
                    try:
                        for content in ('{"value":"ok"}', '{}', '{"value":3}', '{"value":"ok","extra":true}',
                                        '{"value":"bad","value":"ok"}'):
                            response = {"message": {"content": content}, "done": True} if provider == "ollama" else {
                                "choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
                            target = "workbench.postprocess.ollama_request" if provider == "ollama" else "workbench.llm_providers.generate_request"
                            with self.subTest(provider=provider, mode=mode, strict=strict, content=content), \
                                    patch(target, return_value=response) as transport:
                                if content == '{"value":"ok"}':
                                    self.assertEqual(call_json(models=["fixture"], endpoint="http://127.0.0.1:1",
                                        schema=schema, system="test", user="test", temperature=0)[0], json.loads(content))
                                else:
                                    with self.assertRaises(RuntimeError):
                                        call_json(models=["fixture"], endpoint="http://127.0.0.1:1",
                                            schema=schema, system="test", user="test", temperature=0)
                                transport.assert_called_once()
                    finally:
                        CONTEXT.reset(token)

    def test_invalid_structured_output_retries_twice_then_keeps_original_without_summary(self):
        from workbench.postprocess import process_transcript_artifacts
        text = "2026年需要36GiB内存。"
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "transcript.json"
            raw.write_text(json.dumps({"transcription": [{"id": "seg_00000", "text": text,
                "offsets": {"from": 0, "to": 2000}}]}, ensure_ascii=False))
            before = raw.read_bytes()
            config = NextEchoConfig({"postprocess": {"provider": "ollama", "strict_selection": False,
                "preferred_model_id": "qwen3.5:4b-mlx", "fallback_model": "qwen3.5:4b-mlx"}})
            # Domain validation used to accept this numeric ID and silently
            # ignore extra fields; schema validation must reject it first.
            response = {"message": {"content": json.dumps({"segments": [{"id": 0,
                "polished_text": text, "uncertain_terms": [], "change_notes": []}], "extra": True})}, "done": True}
            with patch("workbench.postprocess.ollama_status", return_value={"ready": True,
                    "models": ["qwen3.5:4b-mlx"]}), \
                    patch("workbench.postprocess.ollama_request", return_value=response) as transport:
                result = process_transcript_artifacts(root, profile="balanced", config=config, summarize=False)
            self.assertEqual(transport.call_count, 2)
            self.assertEqual(result["status"], "partial")
            self.assertEqual(raw.read_bytes(), before)
            enhanced = json.loads((root / "transcript.enhanced.json").read_text())
            self.assertEqual(enhanced["enhanced_transcription"][0]["text"], text)
            self.assertFalse((root / "transcript.summary.json").exists())


if __name__ == "__main__":
    unittest.main()
