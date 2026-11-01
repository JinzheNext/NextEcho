from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from workbench.enhancement import enhance_transcript_artifacts, parse_glossary_text


class EnhancementTests(unittest.TestCase):
    def test_parse_glossary_text_supports_rules_and_protected_words(self) -> None:
        rules = parse_glossary_text(
            """
            波客 => 播客
            AGI => 通用人工智能
            protect: NextEcho
            """
        )
        self.assertIn(("波客", "播客"), rules.replacements)
        self.assertIn("NextEcho", rules.protected)

    def test_enhance_transcript_artifacts_writes_parallel_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "transcript.txt").write_text("这是一个波客节目。NextEcho 不应被改坏。\n", encoding="utf-8")
            (root / "transcript.json").write_text(json.dumps({"transcription": []}, ensure_ascii=False), encoding="utf-8")
            payload = enhance_transcript_artifacts(root, user_glossary_text="波客 => 播客\nprotect: NextEcho")
            self.assertTrue(payload["enabled"])
            enhanced_text = (root / "transcript.enhanced.txt").read_text(encoding="utf-8")
            self.assertTrue(enhanced_text.startswith("本文转录由 GitHub 项目：NextEcho 提供支持"))
            self.assertIn("播客节目", enhanced_text)
            self.assertIn("NextEcho", enhanced_text)
            self.assertIn("powered by github.com/JinzheNext/NextEcho - 让播客自动转录进知识库", enhanced_text)
            enhanced_json = json.loads((root / "transcript.enhanced.json").read_text(encoding="utf-8"))
            self.assertIn("enhancement", enhanced_json)
            self.assertTrue((root / "transcript.txt").exists())


if __name__ == "__main__":
    unittest.main()
