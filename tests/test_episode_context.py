from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from workbench.episode_context import crawl_episode_context, html_to_text


class EpisodeContextTests(unittest.TestCase):
    def test_html_to_text_keeps_show_notes_boundaries(self) -> None:
        self.assertEqual(html_to_text("<p>本期嘉宾</p><p>李继刚</p><p>01:23 讨论</p>"), "本期嘉宾\n\n李继刚\n\n01:23 讨论")

    def test_crawl_extracts_next_data_and_top_comments(self) -> None:
        comments = [
            {"id": f"c{index}", "text": f"评论 {index}", "likeCount": index, "author": {"nickname": f"用户{index}"}}
            for index in range(25)
        ]
        next_data = {
            "props": {
                "pageProps": {
                    "episode": {
                        "eid": "episode-1",
                        "title": "E1 对谈",
                        "shownotes": "<p>本期嘉宾</p><p>李继刚</p>",
                        "podcast": {"title": "测试节目", "author": "孟岩"},
                        "commentCount": 25,
                    },
                    "comments": comments,
                }
            }
        }
        html = f'<html><script id="__NEXT_DATA__" type="application/json">{json.dumps(next_data, ensure_ascii=False)}</script></html>'
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            # Exercise the same parser through a deterministic local fixture.
            from unittest.mock import patch

            with patch("workbench.episode_context._fetch_page", return_value=(html, "fixture", "")):
                payload = crawl_episode_context("https://example.test/episode/1", output_dir, crawler="native")

            self.assertEqual(payload["status"], "ready")
            self.assertEqual([item["name"] for item in payload["speaker_candidates"]], ["孟岩", "李继刚"])
            self.assertEqual(len(payload["comments"]), 20)
            self.assertEqual(payload["comments"][0]["like_count"], 24)
            self.assertTrue((output_dir / "source.raw.html").exists())
            self.assertTrue((output_dir / "source.shownotes.md").exists())
            self.assertTrue((output_dir / "source.comments.top20.json").exists())
            self.assertTrue((output_dir / "source.context.json").exists())


if __name__ == "__main__":
    unittest.main()
