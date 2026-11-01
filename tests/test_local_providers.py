import json
import os
import time
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from workbench.llm_providers import CONTEXT, GenerationContext, local_chat_json, provider_configuration
from workbench.contracts import PipelineError

from workbench.config import NextEchoConfig
from workbench.enhancement import default_glossary_rules
from workbench.io_utils import write_json
from workbench.postprocess import ensure_preserved, process_transcript_artifacts


class ProviderTests(unittest.TestCase):
    def test_legacy_unload_has_no_owner_capability(self):
        from workbench.postprocess import unload_model
        with patch("workbench.postprocess.ollama_request") as request:
            with self.assertRaises(PipelineError) as error:
                unload_model("external-model", endpoint="http://127.0.0.1:11434")
        self.assertEqual(error.exception.code, "OWNERSHIP_REQUIRED")
        request.assert_not_called()
        from workbench.llm_providers import ollama_request
        for path in ("/api/generate", "/api/chat"):
            for value in (0, "0", "0s", "0.0ms"):
                with patch("workbench.llm_providers.request_json") as raw, patch("workbench.llm_providers.generate_request") as generate:
                    with self.assertRaises(PipelineError) as error:
                        ollama_request(path, {"model": "external-model", "keep_alive": value}, endpoint="http://127.0.0.1:11434", timeout=1)
                self.assertEqual(error.exception.code, "OWNERSHIP_REQUIRED")
                raw.assert_not_called()
                generate.assert_not_called()

    def setUp(self):
        self.runtime = tempfile.TemporaryDirectory()
        self.environment = patch.dict(os.environ, {"NEXTECHO_RUNTIME_DIR": self.runtime.name})
        self.environment.start()
        self.calls = []
        calls = self.calls
        self.metadata = {"digest": "fake-model-v1"}
        metadata = self.metadata
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_GET(self):
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps({"models": [{"name": "qwen3.5:4b-mlx", "digest": "fake"}],
                    "data": [{"id": "qwen3.8-27b-local", **metadata}]}).encode())
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append(body)
                if body.get("delay"):
                    time.sleep(body["delay"])
                    self.send_response(200)
                    self.end_headers()
                    try:
                        self.wfile.write(b"{}")
                    except BrokenPipeError:
                        pass
                    return
                data = json.loads(body["messages"][-1]["content"].split("\n", 1)[1])
                result = {"segments": [{"id": s["id"], "polished_text": s["text"], "uncertain_terms": [], "change_notes": []} for s in data["segments"]]}
                content = json.dumps(result, ensure_ascii=False)
                payload = {"message": {"content": content}, "eval_count": 32,
                           "choices": [{"message": {"content": content, "reasoning_content": "MUST NOT BE USED"}, "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 100, "completion_tokens": 32}}
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(payload, ensure_ascii=False).encode())
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.environment.stop()
        self.runtime.cleanup()

    def make_item(self, root):
        write_json(root / "transcript.json", {"transcription": [{"id": "seg_00000", "text": "2026年需要36GiB内存。", "offsets": {"from": 0, "to": 2000}}]})

    def test_splash_polish_only_and_context_change_invalidates_chunks(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.make_item(root)
            data = {"postprocess": {"provider": "local-chat-compatible", "endpoint": self.endpoint,
                    "preferred_model_id": "qwen3.8-27b-local", "max_output_tokens": 512, "context_tokens": 16384}}
            first = process_transcript_artifacts(root, profile="quality", config=NextEchoConfig(data), summarize=False)
            self.assertEqual(first["actual_models"], ["qwen3.8-27b-local"])
            self.assertEqual(first["usage"][0]["completion_tokens"], 32)
            self.assertEqual(len(self.calls), 1)
            self.assertFalse((root / "transcript.summary.json").exists())
            self.assertTrue(process_transcript_artifacts(root, profile="quality", config=NextEchoConfig(data), summarize=False)["cached"])
            self.assertEqual(len(self.calls), 1)
            data["postprocess"]["context_tokens"] = 8192
            process_transcript_artifacts(root, profile="quality", config=NextEchoConfig(data), summarize=False)
            self.assertEqual(len(self.calls), 2)
            self.assertNotIn("MUST NOT BE USED", (root / "transcript.enhanced.txt").read_text())
            data["postprocess"]["max_output_tokens"] = 256
            process_transcript_artifacts(root, profile="quality", config=NextEchoConfig(data), summarize=False)
            self.assertEqual(len(self.calls), 3)
            (root / "transcript.enhanced.txt").write_text("corrupted output")
            repaired = process_transcript_artifacts(root, profile="quality", config=NextEchoConfig(data), summarize=False)
            self.assertFalse(repaired.get("cached", False))
            self.assertNotIn("corrupted output", (root / "transcript.enhanced.txt").read_text())
            self.assertEqual(len(self.calls), 3)  # valid chunk checkpoints rebuilt the output


    def test_partial_polish_retries_only_failed_chunk(self):
        from workbench.postprocess import call_json as real_call
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_json(root / "transcript.json", {"transcription": [
                {"id": f"seg_{i:05d}", "text": ("第一段讨论细节。" if i == 0 else "第二段讨论结果。") * 35,
                 "offsets": {"from": i * 1000, "to": (i + 1) * 1000}} for i in range(2)]})
            config = NextEchoConfig({"postprocess": {"provider": "local-chat-compatible", "endpoint": self.endpoint,
                "preferred_model_id": "qwen3.8-27b-local", "polish_max_chars": 400, "polish_target_chars": 200}})
            failures = []
            def fail_second(**kwargs):
                segments = json.loads(kwargs["user"].split("\n", 1)[1])["segments"]
                if segments[0]["id"] == "seg_00001":
                    failures.append(1)
                    raise RuntimeError("fake transient generation error")
                return real_call(**kwargs)
            with patch("workbench.postprocess.call_json", side_effect=fail_second):
                first = process_transcript_artifacts(root, profile="quality", config=config, summarize=False)
            self.assertEqual(first["status"], "partial")
            self.assertEqual(len(failures), 2)
            self.assertEqual(len(self.calls), 1)
            second = process_transcript_artifacts(root, profile="quality", config=config, summarize=False)
            self.assertEqual(second["status"], "ready")
            self.assertEqual(len(self.calls), 2)

    def test_failed_summary_checkpoint_is_retried(self):
        from workbench.postprocess import Segment, summarize_segments
        final = {"title": "项目", "brief_summary": "讨论项目。", "detailed_summary": "讨论项目。",
                 "key_points": [], "chapters": [], "people_organizations_tools": [], "action_items": [],
                 "open_questions": [], "keywords": [], "notable_quotes": []}
        local = {"summary": "讨论项目。", "key_points": [], "segment_ids": ["s"]}
        with tempfile.TemporaryDirectory() as d:
            options = dict(models=["m"], endpoint=self.endpoint, checkpoint_dir=Path(d))
            segments = [Segment("s", "讨论项目。", 0, 1000)]
            with patch("workbench.postprocess.call_json", side_effect=[RuntimeError("local failure"), (final, "m", [])]):
                _, audit = summarize_segments(segments, **options)
                self.assertEqual(audit["status"], "partial")
            with patch("workbench.postprocess.call_json", side_effect=[(local, "m", []), (final, "m", [])]) as generate:
                _, audit = summarize_segments(segments, **options)
                self.assertEqual(audit["status"], "ready")
                self.assertEqual(generate.call_count, 2)

    def test_ollama_native_budget_is_effective(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.make_item(root)
            data = {"postprocess": {"provider": "ollama", "endpoint": self.endpoint,
                    "preferred_model_id": "qwen3.5:4b-mlx", "fallback_model": "qwen3.5:4b-mlx",
                    "context_tokens": 8192, "max_output_tokens": 512}}
            result = process_transcript_artifacts(root, profile="balanced", config=NextEchoConfig(data), summarize=False)
            self.assertEqual(result["status"], "ready")
            self.assertEqual(self.calls[0]["options"]["num_ctx"], 8192)
            self.assertFalse(self.calls[0]["think"])
            self.assertNotIn("format", self.calls[0])
            self.assertIn("Schema", self.calls[0]["messages"][0]["content"])
            data["postprocess"]["structured_mode"] = "native-schema"
            process_transcript_artifacts(root, profile="balanced", config=NextEchoConfig(data), summarize=False)
            self.assertIn("format", self.calls[1])

    def test_chat_rejects_truncation_reasoning_only_and_non_json(self):
        for choice in ({"message": {"content": "{}"}, "finish_reason": "length"},
                       {"message": {"reasoning_content": "{}"}, "finish_reason": "stop"},
                       {"message": {"content": "invalid JSON"}, "finish_reason": "stop"}):
            with patch("workbench.llm_providers.generate_request", return_value={"choices": [choice]}):
                with self.assertRaises(ValueError):
                    local_chat_json(endpoint=self.endpoint, model="test", system="test", user="test", schema={})

    def test_json_envelope_is_exact_and_truncated_ollama_response_is_rejected(self):
        from workbench.llm_providers import decode_model_json
        self.assertEqual(decode_model_json('```json\n{"value":1}\n```'), ({"value": 1}, "json-fence"))
        for text in ('Explanation: {"value":1}', '```json\n{"value":1}\n```\nextra', '```json\n{"value":1', '<think>note</think>{"value":1}', '[1]', '{"value":NaN}'):
            with self.assertRaises(ValueError):
                decode_model_json(text)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.make_item(root)
            config = NextEchoConfig({"postprocess": {"provider": "ollama", "endpoint": self.endpoint,
                "preferred_model_id": "qwen3.5:4b-mlx", "fallback_model": "qwen3.5:4b-mlx"}})
            with patch("workbench.postprocess.ollama_request", return_value={"message": {"content": '{}'}, "done_reason": "length"}) as request:
                result = process_transcript_artifacts(root, profile="balanced", config=config, summarize=False)
            self.assertEqual(result["status"], "partial")
            self.assertEqual(request.call_count, 2)

    def test_strict_provider_validates_required_fields_and_types_before_semantic_guard(self):
        from workbench.llm_providers import validate_json_schema
        from workbench.postprocess import polish_schema
        for row in ({"id": "s", "polished_text": "text"},
                    {"id": 1, "polished_text": "text", "uncertain_terms": [], "change_notes": []},
                    {"id": "s", "polished_text": "text", "uncertain_terms": [1], "change_notes": []},
                    {"id": "s", "polished_text": "text", "uncertain_terms": [], "change_notes": [], "start_ms": True}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                validate_json_schema({"segments": [row]}, polish_schema())

    def test_non_json_transport_does_not_leave_completed_provider_busy(self):
        from workbench.llm_providers import request_json
        from workbench.job_store import JobStore
        import os
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"NEXTECHO_RUNTIME_DIR": d, "NEXTECHO_JOB_ID": "probe"}):
            # A fully received malformed response has finished at the server.
            class Response:
                def __enter__(self): return self
                def __exit__(self, *args): pass
                def read(self, *args): return b"invalid JSON"
            with patch("urllib.request.OpenerDirector.open", return_value=Response()):
                with self.assertRaises(PipelineError):
                    request_json(self.endpoint, "/v1/chat/completions", {})
            store = JobStore()
            store.provider_start(self.endpoint, "next")
            store.provider_end(self.endpoint, completed=True)

    def test_configured_ownership_does_not_authorize_unload_after_failure(self):
        @provider_configuration
        def failing(*, config):
            CONTEXT.get().usage.append({"model": "owned-model"})
            raise ValueError("output failure")
        for ownership in ("external", "owned"):
            with patch("workbench.postprocess.unload_model") as release:
                with self.assertRaises(ValueError):
                    failing(config=NextEchoConfig({"postprocess": {"ownership": ownership}}))
                release.assert_not_called()
        self.assertIsNone(CONTEXT.get())

    def test_provider_configuration_restores_outer_context_on_success(self):
        outer = GenerationContext({"sentinel": True})
        token = CONTEXT.set(outer)
        try:
            @provider_configuration
            def completed(*, config):
                self.assertIsNot(CONTEXT.get(), outer)
                CONTEXT.get().usage.append({"model": "operator-model"})
                return "done"
            with patch("workbench.postprocess.unload_model") as release:
                self.assertEqual(completed(config=NextEchoConfig({"postprocess": {"ownership": "owned"}})), "done")
                release.assert_not_called()
            self.assertIs(CONTEXT.get(), outer)
            self.assertEqual(outer.usage, [])
        finally:
            CONTEXT.reset(token)

    def test_external_timeout_keeps_service_alive_and_lease_blocked(self):
        from workbench.llm_providers import generate_request
        from workbench.job_store import JobStore
        from workbench.resources import ExecutionLease
        token = CONTEXT.set(GenerationContext({"resource_sample_seconds": .05}))
        try:
            with self.assertRaises(PipelineError):
                generate_request(self.endpoint, "/v1/chat/completions", {"model": "fake", "delay": 1.2}, timeout=.5)
            self.assertTrue(self.thread.is_alive())
            with self.assertRaises(PipelineError):
                with ExecutionLease():
                    self.fail("external request was declared idle without confirmation")
            time.sleep(1)
            JobStore().confirm_provider_idle(self.endpoint)
            with ExecutionLease():
                pass
        finally:
            CONTEXT.reset(token)

    def test_unit_negation_and_chinese_quantity_protection(self):
        for before, after in (("需要36GiB内存。", "需要36GB内存。"),
                              ("不支持这个条件。", "支持这个条件。"), ("支付一万元。", "支付一元。"),
                              ("需要1台电脑。", "需要12台电脑。"),
                              ("今天讨论项目。", "今天讨论项目，接着计划投资新业务并安排团队扩张。")):
            with self.assertRaises(ValueError):
                ensure_preserved(before, after, default_glossary_rules())

    def test_cancelled_external_request_keeps_server_alive_and_occupancy_unknown(self):
        from workbench.llm_providers import generate_request
        from workbench.job_store import JobStore
        from workbench.resources import ExecutionLease
        token = CONTEXT.set(GenerationContext({"resource_sample_seconds": .03}))
        try:
            with patch("workbench.job_store.current_cancelled", side_effect=lambda: bool(self.calls)):
                with self.assertRaises(PipelineError) as caught:
                    generate_request(self.endpoint, "/v1/chat/completions", {"model": "fake", "delay": 1}, timeout=5)
            self.assertEqual(caught.exception.code, "CANCELLED")
            self.assertTrue(self.thread.is_alive())
            with self.assertRaises(PipelineError):
                with ExecutionLease():
                    self.fail("cancelled HTTP client does not prove external inference stopped")
            time.sleep(1.1)
            JobStore().confirm_provider_idle(self.endpoint)
        finally:
            CONTEXT.reset(token)

    def test_resource_failure_and_cancel_do_not_become_output_retries(self):
        for code in ("CANCELLED", "RESOURCE_LIMIT", "STAGE_TIMEOUT", "PROVIDER_BUSY_UNKNOWN"):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                self.make_item(root)
                config = NextEchoConfig({"postprocess": {"provider": "local-chat-compatible", "endpoint": self.endpoint,
                    "preferred_model_id": "qwen3.8-27b-local"}})
                with patch("workbench.postprocess.call_json", side_effect=PipelineError(code, "test")) as request:
                    with self.assertRaises(PipelineError) as caught:
                        process_transcript_artifacts(root, profile="quality", config=config, summarize=False)
                self.assertEqual(caught.exception.code, code)
                self.assertEqual(request.call_count, 1)
                self.assertFalse((root / "postprocess.audit.json").exists())

    def test_chat_same_alias_changed_revision_or_unknown_identity_does_not_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.make_item(root)
            config = NextEchoConfig({"postprocess": {"provider": "local-chat-compatible", "endpoint": self.endpoint,
                "preferred_model_id": "qwen3.8-27b-local"}})
            process_transcript_artifacts(root, profile="quality", config=config, summarize=False)
            self.metadata["digest"] = "fake-model-v2"
            changed = process_transcript_artifacts(root, profile="quality", config=config, summarize=False)
            self.assertFalse(changed.get("cached", False))
            self.assertEqual(len(self.calls), 2)
            self.metadata.clear()
            process_transcript_artifacts(root, profile="quality", config=config, summarize=False)
            unknown = process_transcript_artifacts(root, profile="quality", config=config, summarize=False)
            self.assertFalse(unknown.get("cached", False))
            self.assertEqual(len(self.calls), 4)
            self.assertEqual(unknown["model_selection"]["provider_identity"]["cache_identity"], "invocation-only-unverified")
