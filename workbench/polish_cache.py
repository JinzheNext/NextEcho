"""Content identities and portable pointers; audit files are still reverified."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .contracts import PipelineError
from .io_utils import file_hash, value_hash

CANDIDATE_PATH_PROTOCOL = "checkpoint-relative-v1"


def local_candidate_records(rows, checkpoint_dir):
    """Resolve only explicit relative records; legacy absolute paths stay local."""
    if not isinstance(rows, list):
        return None
    try:
        result = []
        root = (checkpoint_dir / "candidates").resolve()
        for row in rows:
            if not isinstance(row, dict):
                return None
            record = dict(row)
            original = Path(row["candidate_path"])
            if not original.is_absolute():
                return None
            protocol = row.get("path_protocol")
            if protocol is not None:
                relative = Path(row["candidate_relative_path"])
                if (protocol != CANDIDATE_PATH_PROTOCOL or relative.is_absolute()
                        or len(relative.parts) != 2 or relative.parts[0] != "candidates"
                        or relative.name in {".", ".."} or relative.name != original.name):
                    return None
                path = (checkpoint_dir / relative).resolve()
                if path != original.resolve() or "candidate_origin_path" in row:
                    record["candidate_origin_path"] = row.get("candidate_origin_path", str(original))
                record["candidate_path"] = str(path)
            else:
                if "candidate_relative_path" in row:
                    return None
                path = original.resolve()
            if (not path.is_relative_to(root) or not path.is_file()
                    or path.stat().st_size > 32 * 1024**2 or file_hash(path) != row["sha256"]):
                return None
            result.append(record)
        return result
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        return None


def cache_input_identity(root, selection, dialogue):
    """Native path-view changes are not changes to text, voices or the model.

    Only explicitly known audit fields are excluded. Current files and their
    full audit hashes are checked before projecting an identity; the original
    selection remains in adoption/audit output, never replaced by this key.
    """
    if selection["layer"] != "acoustic-dialogue" or dialogue.get("backend") != "nemotron-metal-candidate":
        return selection
    from .speaker_service import speaker_artifact_paths

    try:
        paths = speaker_artifact_paths(root, payload=dialogue)
        evidence = dialogue["acoustic_evidence"]
        native = copy.deepcopy(json.loads(Path(paths["native_candidate_json_path"]).read_text()))
        request = copy.deepcopy(json.loads(Path(paths["native_request_json_path"]).read_text()))
        report = copy.deepcopy(json.loads(Path(paths["native_report_json_path"]).read_text()))
        if report["native"] != native or report["status"] != "candidate-probe-complete":
            raise ValueError("Native completion does not match its data")
        if request["audio_sha256"] != dialogue["audio_sha256"]:
            raise ValueError("Native request has another audio identity")
        audio = Path(request["audio_path"])
        if not audio.is_absolute() or not audio.resolve().is_relative_to(root.resolve()) or file_hash(audio) != request["audio_sha256"]:
            raise ValueError("Native audio unavailable or changed")
        if Path(request["output"]).resolve() != Path(paths["native_candidate_json_path"]).resolve():
            raise ValueError("Native request has another output")
        if not Path(request["progress"]).resolve().is_relative_to(root.resolve()):
            raise ValueError("Native progress link escapes item")

        def content_path(path, key, digest):
            if (Path(path).resolve() != Path(paths[key]).resolve() or evidence["hashes"][key] != digest):
                raise ValueError("Native probability/checkpoint disagrees with evidence")
            return digest

        native["probability_path"] = content_path(native["probability_path"], "native_tail_probability_path", native["probability_sha256"])
        archive = native["full_probability_archive"]
        archive["path"] = content_path(archive["path"], "native_probability_path", archive["sha256"])
        for i, row in enumerate(native["checkpoints"]):
            row["path"] = content_path(row["path"], f"native_checkpoint_{i:04d}_path", row["sha256"])
        native.pop("relocations", None)
        native.pop("timings", None)
        request["audio_path"] = request["audio_sha256"]
        request.pop("output")
        request.pop("progress")
        report["native"] = native
        for key in ("resources", "relocation", "resource_measurements_scope"):
            report.pop(key, None)
        semantic = copy.deepcopy(dialogue)
        semantic.pop("resources", None)
        semantic.pop("acoustic_evidence")
        # These known files carry view links or audit telemetry. Every one was
        # checked above/in the canonical reader; text/identity fields remain in
        # semantic, and model/probability/checkpoint identities remain below.
        audit_only = {"transcript.dialogue.json", "transcript.speakers.json", "dialogue.integrity.json"}
        parent_hashes = {key: digest for key, digest in selection["artifact_hashes"].items() if key not in audit_only}
        return {"schema_version": 2, "protocol": "verified-native-content-v1",
                **{key: selection[key] for key in ("policy", "layer", "primary_name")},
                "sha256": value_hash({"dialogue": semantic, "native": native, "request": request,
                                      "report": report, "parent_hashes": parent_hashes})}
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as exc:
        raise PipelineError("CACHE_MISMATCH", "润色Native输入证据无法核验，不能复用或替换为原稿", blocked=True) from exc
