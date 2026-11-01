"""Fake decoder/VAD tests; no inference quality claim or heavy imports."""
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from workbench.local_asr_worker import main


class LocalWorkerTests(unittest.TestCase):
    def run_worker(self, engine, seconds, regions, **options):
        loads = []
        generated = []
        class Audio:
            samplerate, channels = 16000, 1
            def __init__(self, *args): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def __len__(self): return int(seconds * 16000)
            def blocks(self, **kwargs):
                for _ in range(max(1, len(regions))):
                    yield [0.0] * 512
            def seek(self, offset): self.offset = offset
            def read(self, length, **kwargs): return [0.0] * length
        class Vad:
            def __init__(self, *args, **kwargs): self.index, self.rows = 0, []
            def accept_waveform(self, data):
                if self.index < len(regions):
                    start, end = regions[self.index]
                    self.rows.append(types.SimpleNamespace(start=int(start * 16000), samples=[0.0] * int((end - start) * 16000)))
                    self.index += 1
            def empty(self): return not self.rows
            @property
            def front(self): return self.rows[0]
            def pop(self): self.rows.pop(0)
            def flush(self): pass
        class Decoder:
            def __init__(self, **kwargs): loads.append(1)
            def generate(self, **kwargs):
                generated.append(kwargs)
                return [{"text": "正常重复强调"}]
            def create_stream(self):
                return types.SimpleNamespace(accept_waveform=lambda *args: None, result=types.SimpleNamespace(text="正常重复强调"))
            def decode_stream(self, stream): pass
        sf = types.SimpleNamespace(SoundFile=Audio, write=lambda path, *args: Path(path).write_bytes(b"FAKE WAV"))
        sherpa = types.SimpleNamespace(VadModelConfig=lambda: types.SimpleNamespace(silero_vad=types.SimpleNamespace()),
                VoiceActivityDetector=Vad, OfflineRecognizer=types.SimpleNamespace(from_sense_voice=Decoder))
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            request = root / "request.json"
            request.write_text(json.dumps({"schema_version": 1, "engine": engine, "language": "auto", "vad_model": "FAKE", "model": d, "audio": "FAKE", "output": str(root / "result.json"), **options}))
            with patch.dict(sys.modules, {"soundfile": sf, "sherpa_onnx": sherpa, "funasr": types.SimpleNamespace(AutoModel=Decoder)}), patch.object(sys, "argv", ["worker", str(request)]), patch.dict(os.environ, {"NEXTECHO_SUPERVISOR_PID": "0"}):
                main()
            self.generated = generated
            return json.loads((root / "result.json").read_text()), loads

    def test_nano_receives_simplified_language_and_plural_hotwords(self):
        result, _ = self.run_worker("fun-asr-nano", 3, [(0, 2)], language="zh", output_script="simplified", hotwords=["李继刚"])
        self.assertEqual(self.generated[0]["language"], "简体中文")
        self.assertEqual(self.generated[0]["hotwords"], ["李继刚"])
        self.assertNotIn("hotword", self.generated[0])
        self.assertEqual(result["recognition_prompt"]["hotwords"], ["李继刚"])

    def test_one_model_load_for_multiple_windows_and_tail_preserved(self):
        for engine in ("fun-asr-nano", "sensevoice"):
            with self.subTest(engine=engine):
                result, loads = self.run_worker(engine, 31, [(0, 5), (26, 31)])
                self.assertEqual(len(loads), 1)
                self.assertEqual(result["timings"]["model_load_count"], 1)
                self.assertEqual(len(result["timings"]["warm_window_seconds"]), 1)
                self.assertEqual(result["windows"][-1]["raw_end_sample"], 31 * 16000)
                self.assertEqual(len(result["segments"]), 2)
                self.assertTrue(all(w["raw_end_sample"] - w["raw_start_sample"] <= 30 * 16000 for w in result["windows"]))

    def test_long_voiced_region_has_bounded_windows_and_complete_tail(self):
        result, loads = self.run_worker("sensevoice", 61, [(0, 61)])
        self.assertEqual(len(loads), 1)
        self.assertEqual(len(result["windows"]), 3)
        self.assertEqual(result["windows"][-1]["retain_end_sample"], 61 * 16000)
        self.assertEqual(len(result["segments"]), 3)  # repeated emphasis is retained
        self.assertTrue(result["review"])

    def test_overlap_merges_matching_proper_name_without_deleting_new_words(self):
        from workbench.audio_segments import merge_recognized_segments
        rows, review = merge_recognized_segments([
            {"text": "我们今天讨论张三李四。", "start_ms": 0, "end_ms": 30000, "retain_start_ms": 0, "retain_end_ms": 29000},
            {"text": "张三李四，继续介绍。", "start_ms": 28500, "end_ms": 58000, "retain_start_ms": 29000, "retain_end_ms": 57800}])
        self.assertEqual(rows[1]["text"], "继续介绍。")
        self.assertEqual(rows[1]["start_ms"], rows[0]["end_ms"])
        self.assertEqual(review, [])

    def test_short_utterance_not_fragmented_and_silence_does_not_load_asr(self):
        result, loads = self.run_worker("sensevoice", 3, [(0, .7), (1.5, 2.5)])
        self.assertEqual(len(result["windows"]), 1)
        self.assertEqual(result["windows"][0]["raw_start_sample"], 0)
        self.assertEqual(result["windows"][0]["raw_end_sample"], 3 * 16000)
        result, loads = self.run_worker("fun-asr-nano", 3, [])
        self.assertEqual(result["coverage"]["content_state"], "no_speech")
        self.assertEqual(loads, [])
