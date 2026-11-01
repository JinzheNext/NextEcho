"""Durable local jobs and provider ownership. SQLite transactions own state."""
from __future__ import annotations

import json
import contextlib
import hashlib
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

from .contracts import PipelineError
from .delivery import TERMINAL, compact, write_receipt
from .io_utils import file_hash, value_hash, write_json
from .resources import group_running, private_dir, process_alive, process_birth, runtime_dir
from .reading_artifacts import freeze_delivery_context


class JobStore:
    def __init__(self, root: Path | None = None):
        self.root = private_dir((root or runtime_dir()).resolve())
        self.db = self.root / "jobs.sqlite3"
        fd = os.open(self.db, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        self.db.chmod(0o600)
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            version = c.execute("PRAGMA user_version").fetchone()[0]
            if version > 1:
                raise PipelineError("PROTOCOL_ERROR", "运行时数据库版本较新，请升级NextEcho", blocked=True)
            if version == 0:
                tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if tables:
                    # Back up the last committed version through a separate
                    # reader before this writer mutates the schema (WAL-safe).
                    backup = self.root / f"jobs.schema0.{time.time_ns()}.backup.sqlite3"
                    fd = os.open(backup, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
                    os.close(fd)
                    with self.connect() as source, sqlite3.connect(backup) as target:
                        source.backup(target)
                    if "jobs" in tables:
                        columns = {r[1] for r in c.execute("PRAGMA table_info(jobs)")}
                        if columns != {"job_id", "idempotency_key", "request_hash", "status", "data"}:
                            raise PipelineError("PROTOCOL_ERROR", "未知旧任务表结构；已备份，请先迁移运行时", blocked=True)
                c.execute("CREATE TABLE IF NOT EXISTS jobs (job_id TEXT PRIMARY KEY, idempotency_key TEXT UNIQUE, request_hash TEXT, status TEXT NOT NULL, data TEXT NOT NULL)")
                c.execute("CREATE TABLE IF NOT EXISTS providers (endpoint TEXT PRIMARY KEY, job_id TEXT, state TEXT, updated REAL)")
                c.execute("PRAGMA user_version=1")
        with self.connect() as c:
            c.execute("PRAGMA journal_mode=WAL")
        self.db.chmod(0o600)

    @contextlib.contextmanager
    def connect(self):
        c = sqlite3.connect(self.db, timeout=30)
        try:
            c.execute("PRAGMA busy_timeout=30000")
            with c:
                yield c
        finally:
            c.close()

    def submit(self, plan: dict, *, output_dir: Path | None = None, key: str | None = None,
               parent: str = "", resume_from: str = "", submission_context: dict | None = None,
               resume_policy: dict | None = None) -> tuple[dict, bool]:
        # 先用纯路径运算固定caller cwd；长度门仍先于文件系统探测。
        output_dir = Path(os.path.abspath(output_dir.expanduser())) if output_dir is not None else None
        if output_dir is not None:
            compact({"job_id": "0" * 32, "status": "ready", "primary_file": str(output_dir / "result.md"),
                     "receipt_path": str(output_dir / "receipt.json")})
            output_dir = output_dir.resolve()
        hash_input = {"plan": plan, "output_dir": str(output_dir or ""), "resume_from": resume_from}
        if parent or resume_policy is not None:
            hash_input.update(parent_job_id=parent, resume_mode=(resume_policy or {}).get("mode", "exact"))
        request_hash = value_hash(hash_input)
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            if key:
                found = c.execute("SELECT request_hash,data FROM jobs WHERE idempotency_key=?", (key,)).fetchone()
                if found:
                    if found[0] != request_hash:
                        raise PipelineError("INVALID_INPUT", "幂等键已用于不同请求")
                    return self._checked_delivery(json.loads(found[1])), False
            job_id = uuid4().hex
            run = (output_dir or self.root.parent / "transcriptions" / f"run_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_{job_id[:8]}").expanduser().absolute()
            compact({"job_id": job_id, "status": "ready", "primary_file": str(run / "result.md"), "receipt_path": str(run / "receipt.json")})
            run = run.resolve()
            compact({"job_id": job_id, "status": "ready", "primary_file": str(run / "result.md"), "receipt_path": str(run / "receipt.json")})
            if run.exists() and any(run.iterdir()):
                raise PipelineError("CACHE_MISMATCH", "输出目录已有内容，请使用新目录或jobs resume", blocked=True)
            private_dir(run)
            delivery_context = freeze_delivery_context(run, created_at=time.time(),
                parent_dir=Path(resume_from) if resume_from else None)
            private_dir(self.root / "jobs" / job_id)
            payload = {"schema_version": 1, "job_id": job_id, "status": "queued", "stage": "queued",
                       "plan": plan, "run_dir": str(run), "primary_file": "", "receipt_path": str(run / "receipt.json"),
                       "created_at": time.time(), "updated_at": time.time(), "owner_pid": 0, "owner_birth": "",
                       "cancel_requested": False, "parent_job_id": parent, "resume_from": resume_from}
            payload.update(transcribed_on=delivery_context["transcribed_on"], transcription_timezone=delivery_context["timezone"])
            if submission_context is not None:
                payload["submission_context"] = submission_context
            if resume_policy is not None:
                payload["resume_policy"] = resume_policy
                write_json(run / "resume.plan.json", resume_policy)
            payload.update(revision=0, attempt=(self.get(parent, reconcile=False).get("attempt", 1) + 1) if parent else 1)
            write_json(run / "request.json", payload)
            c.execute("INSERT INTO jobs VALUES(?,?,?,?,?)", (job_id, key, request_hash, "queued", json.dumps(payload)))
            return payload, True

    def spawn(self, job_id: str):
        job = self.get(job_id, reconcile=False)
        log_path = self.root / "jobs" / job_id / "worker.log"
        try:
            with log_path.open("ab") as log:
                log_path.chmod(0o600)
                process = subprocess.Popen([sys.executable, "-m", "workbench.job_worker", "--root", str(self.root), "--job", job_id],
                    cwd=Path(__file__).resolve().parent.parent, start_new_session=True, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=log, env={**os.environ, "NEXTECHO_RUNTIME_DIR": str(self.root)})
        except OSError as exc:
            receipt = write_receipt(Path(job["run_dir"]), job_id=job_id, status="failed",
                error_code="SPAWN_FAILED", message="后台任务无法启动，请检查运行时安装及worker.log", plan=job["plan"], stage="prepare")
            self.update(job_id, **{k: v for k, v in receipt.items() if k != "job_id"})
            raise PipelineError("SPAWN_FAILED", receipt["message"]) from exc
        self.update(job_id, owner_pid=process.pid, owner_birth=process_birth(process.pid))
        return process

    def get(self, job_id: str, *, reconcile=True):
        with self.connect() as c:
            row = c.execute("SELECT data FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise PipelineError("INVALID_INPUT", "未找到任务ID")
        data = json.loads(row[0])
        if reconcile and data["status"] in {"ready", "partial"}:
            return self._checked_delivery(data)
        if reconcile and data["status"] not in TERMINAL and (data.get("cancel_requested") or time.time() - data["updated_at"] > 5):
            if not data.get("owner_pid") or not process_alive(data["owner_pid"], data.get("owner_birth", "")):
                if any(process_alive(row["pid"], row["birth"]) or group_running(row["pid"]) for row in data.get("owned_processes", [])):
                    return data
                receipt = Path(data["receipt_path"])
                if receipt.is_file():
                    try:
                        final = json.loads(receipt.read_text())
                        if not isinstance(final, dict):
                            raise ValueError("invalid receipt")
                        if final.get("publication_schema") == 1:
                            from .publication import verify_publication
                            if verify_publication(Path(data["run_dir"]), job_id=job_id) is None:
                                raise ValueError("pipeline generation not committed")
                        hashes = final.get("artifact_hashes") or {}
                        valid_artifacts = isinstance(hashes, dict) and hashes and all(
                            Path(p).is_absolute() and Path(p).resolve().is_relative_to(Path(data["run_dir"]).resolve())
                            and Path(p).is_file() and file_hash(Path(p)) == h for p, h in hashes.items())
                        diagnostic_only = final.get("status") in {"blocked", "failed", "cancelled"} and not final.get("primary_file")
                        valid_plan = final.get("plan_fingerprint", value_hash(data["plan"])) == value_hash(data["plan"])
                        if final.get("job_id") == job_id and valid_plan and final.get("status") in TERMINAL and (valid_artifacts or diagnostic_only):
                            completed = self.update(job_id, expected_revision=data.get("revision", 0), **{k: v for k, v in final.items() if k not in {"job_id", "plan"}})
                            return self._checked_delivery(completed)
                    except (OSError, ValueError, PipelineError):
                        pass
                return self.update(job_id, expected_revision=data.get("revision", 0), receipt_factory=lambda:
                        write_receipt(Path(data["run_dir"]), job_id=job_id, status="cancelled" if data.get("cancel_requested") else "failed",
                            error_code="CANCELLED" if data.get("cancel_requested") else "INTERRUPTED",
                            message="所属执行已结束，可使用jobs resume恢复", plan=data["plan"], stage=data.get("stage", "unknown")))
        return data

    def lookup_key(self, key: str):
        with self.connect() as c:
            row = c.execute("SELECT job_id FROM jobs WHERE idempotency_key=?", (key,)).fetchone()
        if row is None:
            raise PipelineError("REQUEST_NOT_FOUND", "该请求意图尚未提交")
        return self.get(row[0])

    @staticmethod
    def _checked_delivery(data: dict) -> dict:
        """Validate today's delivery without rewriting the historical job."""
        if data["status"] not in {"ready", "partial"}:
            return data
        try:
            root = Path(data["run_dir"]).resolve()
            primary = Path(data.get("primary_file") or "")
            hashes = data.get("artifact_hashes")
            if not primary.is_absolute() or not isinstance(hashes, dict) or not hashes:
                raise ValueError("missing delivery ledger")
            if str(primary.resolve()) not in hashes:
                raise ValueError("primary missing from ledger")
            receipt_path = Path(data["receipt_path"])
            if not receipt_path.resolve().is_relative_to(root):
                raise ValueError("external receipt")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if not isinstance(receipt, dict):
                raise ValueError("invalid receipt object")
            if receipt.get("publication_schema") == 1:
                from .publication import verify_publication
                if verify_publication(root, job_id=data["job_id"]) is None:
                    raise ValueError("missing pipeline commit")
            if (not isinstance(receipt, dict)
                    or receipt.get("job_id") != data["job_id"] or receipt.get("status") != data["status"]
                    or receipt.get("plan_fingerprint") != value_hash(data["plan"])
                    or receipt.get("primary_file") != str(primary)
                    or receipt.get("artifact_hashes") != hashes):
                raise ValueError("receipt identity or ledger changed")
            for name, expected in hashes.items():
                path = Path(name)
                if not path.is_absolute() or not path.resolve().is_relative_to(root):
                    raise ValueError("external artifact")
                if not path.is_file() or file_hash(path) != expected:
                    raise ValueError("artifact missing or changed")
        except (OSError, ValueError, TypeError, KeyError, PipelineError):
            # This is a query projection. The execution record, receipt and
            # earlier terminal state remain immutable and available to resume.
            return {**data, "status": "blocked", "execution_status": data["status"],
                    "delivery_integrity": "invalid", "primary_file": "",
                    "error_code": "CACHE_MISMATCH",
                    "message": "执行记录已保留，但交付文件缺失、改变或无法校验，请恢复校验后交付。",
                    "next_action": f"使用jobs resume {data['job_id']}创建新attempt，或恢复原始交付文件。"}
        return {**data, "execution_status": data["status"], "delivery_integrity": "verified"}

    def update(self, job_id: str, *, expected_revision=None, receipt_factory=None, **fields):
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT data FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if not row:
                raise PipelineError("INVALID_INPUT", "未找到任务ID")
            data = json.loads(row[0])
            if data["status"] in TERMINAL:
                return data
            if expected_revision is not None and expected_revision != data.get("revision", 0):
                return data
            if receipt_factory:
                fields.update({k: v for k, v in receipt_factory().items() if k not in {"job_id", "plan"}})
            data.update(fields)
            data["revision"] = data.get("revision", 0) + 1
            data["updated_at"] = time.time()
            c.execute("UPDATE jobs SET status=?,data=? WHERE job_id=?", (data["status"], json.dumps(data), job_id))
            return data

    def claim(self, job_id):
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT data FROM jobs WHERE job_id=? AND status='queued'", (job_id,)).fetchone()
            if not row:
                return None
            data = json.loads(row[0])
            if data.get("owner_pid") not in {0, os.getpid()}:
                return None
            data.update(status="running", owner_pid=os.getpid(), owner_birth=process_birth(os.getpid()), updated_at=time.time())
            data["revision"] = data.get("revision", 0) + 1
            c.execute("UPDATE jobs SET status='running',data=? WHERE job_id=?", (json.dumps(data), job_id))
            return data

    def cancelled(self, job_id):
        return self.get(job_id, reconcile=False).get("cancel_requested", False)

    def register_owned(self, job_id, pid, birth):
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT data FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if not row:
                raise PipelineError("INVALID_INPUT", "所属任务不存在")
            data = json.loads(row[0])
            if data["status"] in TERMINAL:
                raise PipelineError("CANCELLED", "任务已结束")
            data.setdefault("owned_processes", []).append({"pid": pid, "birth": birth})
            data["revision"] = data.get("revision", 0) + 1
            c.execute("UPDATE jobs SET data=? WHERE job_id=?", (json.dumps(data), job_id))

    def provider_start(self, endpoint: str, job_id: str):
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            previous = c.execute("SELECT state FROM providers WHERE endpoint=?", (endpoint,)).fetchone()
            if previous and previous[0] != "completed":
                raise PipelineError("PROVIDER_BUSY_UNKNOWN", "上次请求是否结束尚无法确认，请检查本地服务状态后重试", blocked=True)
            c.execute("INSERT OR REPLACE INTO providers VALUES(?,?,?,?)", (endpoint, job_id, "active", time.time()))

    def ensure_providers_idle(self):
        with self.connect() as c:
            busy = c.execute("SELECT 1 FROM providers WHERE state!='completed' LIMIT 1").fetchone()
        if busy:
            raise PipelineError("PROVIDER_BUSY_UNKNOWN", "本地服务仍有活动或状态未确认的请求，请先确认旧推理已结束", blocked=True)

    def provider_end(self, endpoint: str, *, completed: bool):
        with self.connect() as c:
            c.execute("UPDATE providers SET state=?,updated=? WHERE endpoint=?", ("completed" if completed else "unknown", time.time(), endpoint))

    def confirm_provider_idle(self, endpoint: str):
        with self.connect() as c:
            row = c.execute("SELECT job_id,state FROM providers WHERE endpoint=?", (endpoint,)).fetchone()
        if row:
            if row[0].startswith("adhoc:"):
                _, pid, birth_hash = row[0].split(":")
                live = hashlib.sha256(process_birth(int(pid)).encode()).hexdigest()[:20] == birth_hash
                if row[1] == "active" and live:
                    raise PipelineError("INVALID_INPUT", "旧入口请求仍在运行，不能解除服务占用")
            else:
                owner = self.get(row[0])
                if owner["status"] not in TERMINAL:
                    raise PipelineError("INVALID_INPUT", "关联任务仍在运行，不能解除服务占用")
        self.provider_end(endpoint, completed=True)


def current_cancelled():
    job_id = os.environ.get("NEXTECHO_JOB_ID")
    return bool(job_id and JobStore().cancelled(job_id))
