from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from workbench.cli import _apply_config_defaults, build_parser
from workbench.config import DEFAULTS, NextEchoConfig, load_config


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("legacy_link_flags", ROOT / "scripts/transcribe_link.py")
LINK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LINK)


class LegacyFlagTests(unittest.TestCase):
    def test_speaker_count_bounds_are_not_hidden_defaults(self):
        self.assertNotIn("min_speakers", DEFAULTS["diarization"])
        self.assertNotIn("max_speakers", DEFAULTS["diarization"])
        with patch("workbench.cli.load_config", return_value=NextEchoConfig(DEFAULTS)):
            args = build_parser().parse_args(["speaker-transcript", "audio.wav"])
        _apply_config_defaults(args, NextEchoConfig(DEFAULTS), ["speaker-transcript", "audio.wav"])
        self.assertIsNone(args.min_speakers)
        self.assertIsNone(args.max_speakers)

    def test_main_cli_accepted_abbreviations_remain_explicit(self):
        config = NextEchoConfig({"asr": {"language": "zh", "engine": "whisper-cpp"},
            "postprocess": {"profile": "quality"}, "diarization": {
                "default_mode": "required", "backend": "configured", "min_speakers": 2, "max_speakers": 6}})
        argv = ["transcribe", "test.wav", "--lang=en", "--asr-eng=sensevoice",
                "--postprocess-pro=off", "--speaker-m=off", "--speaker-b=explicit",
                "--min-s=1", "--max-spe=3"]
        args = build_parser().parse_args(argv)
        _apply_config_defaults(args, config, argv)
        self.assertEqual((args.language, args.asr_engine, args.postprocess_profile, args.speaker_mode,
                          args.speaker_backend, args.min_speakers, args.max_speakers),
                         ("en", "sensevoice", "off", "off", "explicit", 1, 3))

    def test_terminator_positional_values_do_not_count_as_options(self):
        argv = ["transcribe", "--", "--language=en", "--postprocess-profile=off"]
        args = build_parser().parse_args(argv)
        _apply_config_defaults(args, NextEchoConfig({"asr": {"language": "ja"},
            "postprocess": {"profile": "balanced"}}), argv)
        self.assertEqual(args.sources, ["--language=en", "--postprocess-profile=off"])
        self.assertEqual((args.language, args.postprocess_profile), ("ja", "balanced"))

    def invoke(self, root, options, *, stdout=None, returncode=0):
        output = root / "result"
        output.mkdir(exist_ok=True)
        completed = subprocess.CompletedProcess([], returncode,
            stdout if stdout is not None else json.dumps({"output_dir": str(output)}), "child diagnostic\n")
        out, err = io.StringIO(), io.StringIO()
        with patch.object(LINK.subprocess, "run", return_value=completed) as child, \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = LINK.legacy_main(options)
        return code, child, out.getvalue(), err.getvalue(), output

    def write_config(self, root):
        path = root / "selected config.toml"
        path.write_text('[asr]\nlanguage="zh"\nengine="whisper-cpp"\n'
                        '[postprocess]\nprofile="quality"\n[diarization]\n'
                        'default_mode="required"\nbackend="configured"\nmin_speakers=2\nmax_speakers=6\n')
        return path

    def test_wrapper_config_position_and_explicit_argument_forms(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            config = self.write_config(root)
            for engine in ("fun-asr-nano", "sensevoice"):
                values = [("profile", "off"), ("language", "en"), ("asr-engine", engine),
                          ("speaker-mode", "off"), ("speaker-backend", "explicit"),
                          ("min-speakers", "1"), ("max-speakers", "3")]
                for form in ("equals", "separate", "abbreviated"):
                    flags = ([f"--{key}={value}" for key, value in values] if form == "equals" else
                             [part for key, value in values for part in (f"--{key}", value)] if form == "separate" else
                             ["--prof=off", "--lang=en", f"--asr-eng={engine}", "--speaker-m=off",
                              "--speaker-b=explicit", "--min-s=1", "--max-spe=3"])
                    for before in (True, False):
                        with self.subTest(engine=engine, form=form, config_before=before):
                            selected = ["--config", str(config)]
                            args = [*selected, "https://example.org/episode", *flags] if before else [
                                "https://example.org/episode", *flags, *selected]
                            with patch("workbench.config.load_config", wraps=load_config) as loader:
                                code, child, out, _, output = self.invoke(root, args)
                            self.assertEqual(code, 0)
                            self.assertEqual(loader.call_count, 1)
                            self.assertEqual(out, str(output) + "\n")
                            command = child.call_args.args[0]
                            for key, value in values:
                                flag = "--postprocess-profile" if key == "profile" else f"--{key}"
                                self.assertEqual(command[command.index(flag) + 1], value)
                            request = json.loads((output / "skill_request.json").read_text())
                            self.assertEqual((request["profile"], request["speaker_mode"], request["asr_engine"]),
                                             ("off", "off", engine))

    def test_wrapper_omitted_defaults_and_caller_relative_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            config = self.write_config(root)
            previous = Path.cwd()
            try:
                os.chdir(root)
                code, child, _, _, _ = self.invoke(root, ["https://example.org/episode", "--config", config.name,
                    "--output-dir", "new output", "--glossary", "terms.txt"])
            finally:
                os.chdir(previous)
            self.assertEqual(code, 0)
            command = child.call_args.args[0]
            self.assertEqual(child.call_args.kwargs["cwd"], ROOT)
            for flag, value in (("--postprocess-profile", "quality"), ("--speaker-mode", "required"),
                                ("--config", str(config)), ("--output-dir", str(root / "new output")),
                                ("--glossary", str(root / "terms.txt"))):
                self.assertEqual(command[command.index(flag) + 1], value)

    def test_wrapper_rejects_invalid_success_protocol_without_traceback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for response in ("bad json", "[]", "{}", '{"output_dir":null}', '{"output_dir":3}',
                             json.dumps({"output_dir": str(root / "missing")})):
                with self.subTest(response=response):
                    code, _, out, err, output = self.invoke(root, ["https://example.org/episode"], stdout=response)
                    self.assertEqual(code, 3)
                    self.assertEqual(out, "")
                    self.assertIn("PROTOCOL_ERROR", err)
                    self.assertFalse((output / "skill_request.json").exists())

    def test_wrapper_nonzero_child_preserves_exit_and_diagnostic(self):
        with tempfile.TemporaryDirectory() as temp:
            code, _, out, err, output = self.invoke(Path(temp), ["https://example.org/episode"],
                stdout="child failure\n", returncode=4)
            self.assertEqual((code, out), (4, ""))
            self.assertIn("child diagnostic\nchild failure\n", err)
            self.assertFalse((output / "skill_request.json").exists())

    def test_wrapper_does_not_consume_flags_after_terminator(self):
        with patch("workbench.cli.main", return_value=0) as runtime, patch.object(LINK, "legacy_main") as legacy:
            self.assertEqual(LINK.main(["--", "--legacy", "--json"]), 0)
        legacy.assert_not_called()
        runtime.assert_called_once_with(["run", "--", "--legacy", "--json"])

    def test_json_alias_stays_before_terminator(self):
        with patch("workbench.cli.main", return_value=0) as runtime:
            LINK.main(["--json", "--", "--legacy"])
        runtime.assert_called_once_with(["run", "--output", "json", "--", "--legacy"])


if __name__ == "__main__":
    unittest.main()
