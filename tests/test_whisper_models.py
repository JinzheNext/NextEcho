"""坏模型/有限锁/原子准备回归；toy权重不代表真实ASR质量。"""
from __future__ import annotations

import hashlib
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from workbench.contracts import PipelineError
from workbench.whisper_download import ModelDownloadLock, _birth_hash, prepare_model
from workbench.whisper_download_worker import HTTPSOnlyRedirect, download
from workbench.whisper_models import cached_model, model_spec, registry, validate_whisper_model


def toy_model():
    # Independent miniature native-format fixture: 1 encoder + 1 decoder,
    # all 50 tensors, F32. It has no learned speech-recognition capability.
    data = bytearray(struct.pack("<12i", 0x67676D6C, 1024, 2, 4, 1, 1, 2, 4, 1, 1, 80, 0))
    data += struct.pack("<2i", 80, 1) + b"\0" * 320 + struct.pack("<i", 1024) + b"\0" * 4096
    rows = [("encoder.positional_embedding", (4, 2)), ("encoder.conv1.weight", (3, 80, 4)),
            ("encoder.conv1.bias", (1, 4)), ("encoder.conv2.weight", (3, 4, 4)), ("encoder.conv2.bias", (1, 4)),
            ("encoder.ln_post.weight", (4,)), ("encoder.ln_post.bias", (4,)),
            ("decoder.positional_embedding", (4, 2)), ("decoder.token_embedding.weight", (4, 1024)),
            ("decoder.ln.weight", (4,)), ("decoder.ln.bias", (4,))]
    for kind in ("encoder", "decoder"):
        prefix = kind + ".blocks.0."
        for ln in (["attn_ln", "mlp_ln", "cross_attn_ln"] if kind == "decoder" else ["attn_ln", "mlp_ln"]):
            for suffix in ("weight", "bias"):
                rows.append((prefix + ln + "." + suffix, (4,)))
        for attention in (["attn", "cross_attn"] if kind == "decoder" else ["attn"]):
            for projection in ("query", "key", "value", "out"):
                rows.append((prefix + attention + "." + projection + ".weight", (4, 4)))
                if projection != "key":
                    rows.append((prefix + attention + "." + projection + ".bias", (4,)))
        rows.extend([(prefix + "mlp.0.weight", (4, 16)), (prefix + "mlp.0.bias", (16,)),
                     (prefix + "mlp.2.weight", (16, 4)), (prefix + "mlp.2.bias", (4,))])
    import math
    for name, shape in rows:
        encoded = name.encode()
        data += struct.pack("<3i", len(shape), len(encoded), 0) + struct.pack("<" + "i" * len(shape), *shape) + encoded
        data += b"\0" * (4 * math.prod(shape))
    return bytes(data)


class WhisperModelTests(unittest.TestCase):
    def test_registry_pins_all_official_names_and_rejects_unknown_paths(self):
        self.assertEqual(len(registry()["models"]), 33)
        spec = model_spec("tiny")
        self.assertEqual(spec["bytes"], 77691713)
        self.assertIn("5359861c739e955e79d9a303bcbc70fb988958b1", spec["url"])
        self.assertNotIn("/main/", spec["url"])
        for name in ("../tiny", "unknown", "", "ggml-tiny.bin"):
            with self.assertRaises(PipelineError):
                model_spec(name)

    def test_complete_custom_model_has_identity_without_quality_claim(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "custom.bin"
            path.write_bytes(toy_model())
            identity = validate_whisper_model(path)
            self.assertEqual(identity["tensor_count"], 50)
            self.assertEqual(identity["sha256"], hashlib.sha256(toy_model()).hexdigest())
            self.assertEqual(identity["identity"], "user-supplied")
            self.assertEqual(identity["license_declaration"], "unknown")
            self.assertFalse(identity["quality_verified"])
            self.assertEqual(cached_model(str(path), []), path.resolve())
            with self.assertRaises(PipelineError):
                validate_whisper_model(path, name="tiny")

    def test_corrupt_empty_truncated_headers_tensors_and_directory_are_not_ready(self):
        raw = toy_model()
        bad = [b"", b"model", b"<html>network login</html>" * 8, raw[:4480], raw[:-1], raw + b"x",
               raw.replace(b"decoder.ln.bias", b"decoder.ln.biax"), raw[:48] + b"\xff" * 8 + raw[56:]]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.bin"
            for value in bad:
                path.write_bytes(value)
                with self.assertRaises(PipelineError) as error:
                    validate_whisper_model(path)
                self.assertEqual(error.exception.code, "MODEL_INVALID")
            with self.assertRaises(PipelineError):
                validate_whisper_model(Path(d))
            if os.name == "posix":
                fifo = Path(d) / "fifo.bin"
                os.mkfifo(fifo)
                with self.assertRaises(PipelineError):
                    validate_whisper_model(fifo)

    def test_missing_local_path_does_not_become_remote_name(self):
        for path in ("/missing/weights.bin", "relative/model.bin", "weights.bin"):
            with self.assertRaises(PipelineError) as error:
                cached_model(path, [])
            self.assertEqual(error.exception.code, "MODEL_MISSING")

    def test_named_cache_and_doctor_reject_nonempty_corruption(self):
        from workbench.doctor import check_model
        from workbench.transcription import _find_existing_model, resolve_whisper_model, _bounded_whisper
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"TRANSCRIBE_MODEL_DIR": d, "WHISPER_MODEL_DIR": d}):
            root = Path(d)
            (root / "ggml-tiny.bin").write_bytes(b"not-empty")
            with self.assertRaises(PipelineError):
                _find_existing_model("tiny", root)
            self.assertFalse(check_model("tiny").ok)
            with patch("workbench.transcription.ensure_whisper_model") as prepare:
                with self.assertRaises(PipelineError):
                    resolve_whisper_model(str(root))
                prepare.assert_not_called()
            with patch("workbench.resources.run_owned") as run:
                with self.assertRaises(PipelineError):
                    _bounded_whisper(["fake"], root / "ggml-tiny.bin", root)
                run.assert_not_called()


@unittest.skipUnless(os.name == "posix", "POSIX model lock ownership")
class ModelLockTests(unittest.TestCase):
    def test_empty_or_unknown_legacy_lock_returns_action_without_deleting(self):
        for content in ("", "old pid", "{}", '{"schema_version":2}'):
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                sentinel = root / ".ggml-tiny.lock"
                sentinel.write_text(content)
                with self.assertRaises(PipelineError) as error:
                    with ModelDownloadLock(root, "tiny", timeout=0.15):
                        self.fail("stole unrecognized lock")
                self.assertEqual(error.exception.code, "MODEL_LOCK_LEGACY")
                self.assertEqual(sentinel.read_text(), content)

    def test_active_os_guard_has_finite_wait_and_persistent_inode(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with ModelDownloadLock(root, "tiny") as first:
                original = first.guard.stat().st_ino
                with self.assertRaises(PipelineError) as error:
                    with ModelDownloadLock(root, "tiny", timeout=0.15):
                        self.fail("stole live OS lock")
                self.assertEqual(error.exception.code, "MODEL_BUSY")
                owner = json.loads(first.path.read_text())
                self.assertNotIn("command", owner)
                self.assertEqual(len(owner["birth_sha256"]), 64)
            self.assertFalse((root / ".ggml-tiny.lock").exists())
            with ModelDownloadLock(root, "tiny") as second:
                self.assertEqual(second.guard.stat().st_ino, original)

    def test_known_dead_owner_is_recovered_but_live_owner_not_stolen(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(20)"])
            try:
                owner = {"schema_version": 2, "pid": child.pid, "birth_sha256": _birth_hash(child.pid),
                         "nonce": "a" * 32, "name": "tiny"}
                sentinel = root / ".ggml-tiny.lock"
                sentinel.write_text(json.dumps(owner))
                with self.assertRaises(PipelineError) as error:
                    with ModelDownloadLock(root, "tiny", timeout=0.15):
                        self.fail("stole live owner")
                self.assertEqual(error.exception.code, "MODEL_BUSY")
                self.assertIsNone(child.poll())
            finally:
                child.terminate()
                child.wait(timeout=3)
            with ModelDownloadLock(root, "tiny"):
                self.assertEqual(json.loads(sentinel.read_text())["pid"], os.getpid())

    def test_wait_honors_cancellation_and_never_deletes_other_sentinel(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with ModelDownloadLock(root, "tiny") as first:
                body = first.path.read_bytes()
                with patch("workbench.whisper_download.current_cancelled", return_value=True):
                    with self.assertRaises(PipelineError) as error:
                        with ModelDownloadLock(root, "tiny", timeout=0.15):
                            self.fail("ignored cancellation")
                self.assertEqual(error.exception.code, "CANCELLED")
                self.assertEqual(first.path.read_bytes(), body)


class AtomicPreparationTests(unittest.TestCase):
    def fixture(self, root):
        catalog = registry()
        data = toy_model()
        catalog["models"]["tiny"].update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        path = root / "test-registry.json"
        path.write_text(json.dumps(catalog))
        return patch("workbench.whisper_models.REGISTRY", path)

    def context(self, root):
        return patch.dict(os.environ, {"TRANSCRIBE_MODEL_DIR": str(root / "models"), "WHISPER_MODEL_DIR": str(root / "models"),
                                     "NEXTECHO_RUNTIME_DIR": str(root / "runtime"), "NEXTECHO_JOB_ID": ""})

    def test_same_size_valid_format_wrong_digest_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            path = root / "renamed.bin"
            data = toy_model()
            path.write_bytes(data[:-1] + b"\x01")
            with self.fixture(root):
                self.assertEqual(validate_whisper_model(path)["identity"], "user-supplied")
                with self.assertRaises(PipelineError) as error:
                    validate_whisper_model(path, name="tiny")
                self.assertEqual(error.exception.code, "MODEL_INVALID")

    def test_explicit_download_cli_uses_repair_entry_for_known_name(self):
        from argparse import Namespace
        from contextlib import redirect_stdout
        from workbench.cli import cmd_download_model
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.bin"
            path.write_bytes(toy_model())
            stdout = io.StringIO()
            with patch("workbench.transcription.ensure_whisper_model", return_value=path) as prepare, patch("workbench.cli.resolve_whisper_model", side_effect=AssertionError("ordinary read resolver")), redirect_stdout(stdout):
                self.assertEqual(cmd_download_model(Namespace(model="tiny", json=True)), 0)
            prepare.assert_called_once_with("tiny")
            self.assertEqual(json.loads(stdout.getvalue())["provenance"]["sha256"], hashlib.sha256(toy_model()).hexdigest())

    def test_success_is_verified_atomic_and_old_invalid_bytes_retained(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            target = root / "models/ggml-tiny.bin"
            target.parent.mkdir()
            target.write_bytes(b"bad old cache")
            def transport(spec, part):
                self.assertEqual(target.read_bytes(), b"bad old cache")
                part.write_bytes(toy_model())
            with self.fixture(root), self.context(root), patch("workbench.whisper_download._download_file", side_effect=transport) as run:
                self.assertEqual(prepare_model("tiny", target.parent, []), target.resolve())
                self.assertEqual(prepare_model("tiny", target.parent, []), target.resolve())
                self.assertEqual(run.call_count, 1)
            self.assertEqual(target.read_bytes(), toy_model())
            self.assertEqual(next(target.parent.glob("*.invalid-*.bin")).read_bytes(), b"bad old cache")
            self.assertEqual(list(target.parent.glob("*.part")), [])
            self.assertFalse((target.parent / ".ggml-tiny.lock").exists())

    def test_failed_or_corrupt_download_does_not_replace_old_target(self):
        for failure in ("corrupt", "cancel", "network"):
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                target = root / "models/ggml-tiny.bin"
                target.parent.mkdir()
                target.write_bytes(b"old bytes")
                def transport(spec, part):
                    part.write_bytes(b"partial transport")
                    if failure != "corrupt":
                        raise PipelineError("CANCELLED" if failure == "cancel" else "MODEL_DOWNLOAD_FAILED", failure)
                with self.fixture(root), self.context(root), patch("workbench.whisper_download._download_file", side_effect=transport):
                    with self.assertRaises(PipelineError):
                        prepare_model("tiny", target.parent, [])
                self.assertEqual(target.read_bytes(), b"old bytes")
                self.assertEqual(list(target.parent.glob("*.part")), [])
                self.assertEqual(list(target.parent.glob("*.invalid-*.bin")), [])
                self.assertFalse((target.parent / ".ggml-tiny.lock").exists())

    def test_directory_or_symlink_target_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            target = root / "models/ggml-tiny.bin"
            target.mkdir(parents=True)
            with self.fixture(root), self.context(root), patch("workbench.whisper_download._download_file") as run:
                with self.assertRaises(PipelineError):
                    prepare_model("tiny", target.parent, [])
                run.assert_not_called()
                self.assertTrue(target.is_dir())
                target.rmdir()
                (root / "invalid.bin").write_bytes(b"invalid")
                target.symlink_to(root / "invalid.bin")
                with self.assertRaises(PipelineError):
                    prepare_model("tiny", target.parent, [])
                run.assert_not_called()
                self.assertTrue(target.is_symlink())

    @unittest.skipUnless(os.name == "posix", "shared POSIX lease")
    def test_concurrent_preparation_only_transports_once(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            calls, errors, results = [], [], []
            def transport(spec, part):
                calls.append(1)
                time.sleep(0.1)
                part.write_bytes(toy_model())
            def invoke():
                try:
                    results.append(prepare_model("tiny", root / "models", [], lock_timeout=3))
                except BaseException as exc:
                    errors.append(exc)
            with self.fixture(root), self.context(root), patch("workbench.whisper_download._download_file", side_effect=transport):
                threads = [threading.Thread(target=invoke) for _ in range(3)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=8)
                self.assertFalse(any(t.is_alive() for t in threads))
            self.assertEqual(errors, [])
            self.assertEqual(len(results), 3)
            self.assertEqual(calls, [1])


class TransportTests(unittest.TestCase):
    def test_stream_checks_size_and_digest_before_success(self):
        from types import SimpleNamespace
        class Body(io.BytesIO):
            headers = {}
        raw = b"x" * 100
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "part"
            spec = {"url": "https://example.test/model", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            for body, expected in ((raw, True), (raw[:-1], False), (raw + b"y", False), (b"z" * 100, False)):
                opener = SimpleNamespace(open=lambda *a, **kw: Body(body))
                if expected:
                    self.assertEqual(download(spec, path, opener=opener)["sha256"], spec["sha256"])
                else:
                    with self.assertRaises(ValueError):
                        download(spec, path, opener=opener)
            with self.assertRaises(ValueError):
                download({**spec, "url": "http://example.test"}, path)
            with self.assertRaises(ValueError):
                HTTPSOnlyRedirect().redirect_request(None, None, 302, "", {}, "http://example.test")


if __name__ == "__main__":
    unittest.main()
