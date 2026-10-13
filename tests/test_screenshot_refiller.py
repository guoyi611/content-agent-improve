import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.nodes.screenshot_refiller import (
    _discover_candidates,
    _pick_refill_insert_points,
    screenshot_refiller_node,
)


class ScreenshotRefillerTests(unittest.TestCase):
    @patch("agent.nodes.screenshot_refiller.preflight_screenshot_url")
    @patch("agent.nodes.screenshot_refiller.search")
    def test_discovery_keeps_only_real_search_results(self, search, preflight):
        search.return_value = [
            {"url": "https://docs.example.com/guides/a", "title": "Official guide A"},
            {"url": "https://docs.example.com/missing", "title": "Missing"},
            {"url": "https://docs.example.com/guides/b", "title": "Official guide B"},
        ]
        preflight.side_effect = lambda url: "missing" not in url
        found = _discover_candidates("Example 产品教程", [(1, "功能")], set())
        self.assertEqual(
            {"https://docs.example.com/guides/a", "https://docs.example.com/guides/b"},
            {url for url, _ in found},
        )

    @patch("agent.nodes.screenshot_refiller.get_llm")
    @patch("agent.nodes.screenshot_refiller._discover_candidates")
    def test_refiller_rejects_invented_url_and_uses_verified_pool(self, discover, get_llm):
        discover.return_value = [
            ("https://docs.example.com/guides/a", "功能 A"),
            ("https://docs.example.com/guides/b", "功能 B"),
        ]
        get_llm.return_value.invoke.return_value = SimpleNamespace(
            content="[REPLACEMENT: 1 | https://invented.example/not-real | 编造页面]"
        )
        state = {
            "topic": "Example 产品教程",
            "draft": (
                "[SCREENSHOT: https://old.example/dead-a, 位置一]\n"
                "[SCREENSHOT: https://old.example/dead-b, 位置二]"
            ),
            "screenshot_attempted_urls": [],
            "screenshot_source_urls": [],
            "log": [],
        }
        result = screenshot_refiller_node(state)
        self.assertNotIn("invented.example", result["draft"])
        self.assertIn("https://docs.example.com/guides/a", result["draft"])
        self.assertIn("https://docs.example.com/guides/b", result["draft"])


class PickRefillInsertPointsTests(unittest.TestCase):
    """narrative_design 改动 4：refiller 按 image_placeholders 元数据找未覆盖
    ## heading 补图。"""

    def _draft_with_headings(self):
        return (
            "# 标题\n\n"
            "开篇段。\n"
            "## 章节 A\n\n"
            "A 正文。\n"
            "## 章节 B\n\n"
            "B 正文。\n"
            "## 章节 C\n\n"
            "C 正文。\n"
        )

    def test_no_metadata_falls_back_to_first_n_headings(self):
        """缺 image_placeholders → 降级到前 N 个 heading（保留旧行为）。"""
        draft = self._draft_with_headings()
        headings = list(__import__("re").finditer(r"(?m)^##\s+.+$", draft))
        points = _pick_refill_insert_points(draft, None, 2)
        self.assertEqual(points, [headings[0].end(), headings[1].end()])

    def test_empty_metadata_falls_back_to_first_n_headings(self):
        draft = self._draft_with_headings()
        points = _pick_refill_insert_points(draft, [], 2)
        # 空列表视为缺元数据 → 降级
        self.assertEqual(len(points), 2)

    def test_skips_already_covered_headings(self):
        """已有占位符覆盖 A、B，补图应跳到 C（以及更多时继续）。"""
        draft = self._draft_with_headings()
        headings = list(__import__("re").finditer(r"(?m)^##\s+.+$", draft))
        # A、B 已被占位符覆盖，C 未覆盖
        covered_draft = (
            "## 章节 A\n\n"
            "A 正文。\n"
            "[SCREENSHOT: https://a, A]\n"
            "## 章节 B\n\n"
            "B 正文。\n"
            "[SCREENSHOT: https://b, B]\n"
            "## 章节 C\n\n"
            "C 正文。\n"
        )
        metadata = [
            {"placeholder": "[SCREENSHOT: https://a, A]", "heading": "## 章节 A"},
            {"placeholder": "[SCREENSHOT: https://b, B]", "heading": "## 章节 B"},
        ]
        points = _pick_refill_insert_points(covered_draft, metadata, 2)
        # 仅 C 未覆盖 → 第一个插入点；need=2 但只有一个空位 → 只返 1 个
        self.assertEqual(len(points), 1)
        # 位置对应 C 的结尾（在 covered_draft 里定位）
        c_offset = covered_draft.index("## 章节 C")
        self.assertGreater(points[0], c_offset)

    def test_dropped_placeholders_do_not_count_as_covered(self):
        """已被 _drop_unanchored_screenshots 删除的占位符不算覆盖。"""
        draft = self._draft_with_headings()
        # 元数据声称 A 已有占位符，但 draft 里没有该占位符（已被 drop）
        metadata = [
            {"placeholder": "[SCREENSHOT: https://a, A]", "heading": "## 章节 A"},
        ]
        points = _pick_refill_insert_points(draft, metadata, 1)
        # A 实际没占位符 → 应作为插入点
        headings = list(__import__("re").finditer(r"(?m)^##\s+.+$", draft))
        self.assertEqual(points, [headings[0].end()])

    def test_all_headings_covered_returns_empty(self):
        """所有 heading 都有占位符 → 无插入点，调用方把多余候选追加到文末。"""
        draft = (
            "## 章节 A\n\n"
            "A 正文。\n"
            "[SCREENSHOT: https://a, A]\n"
            "## 章节 B\n\n"
            "B 正文。\n"
            "[SCREENSHOT: https://b, B]\n"
            "## 章节 C\n\n"
            "C 正文。\n"
            "[SCREENSHOT: https://c, C]\n"
        )
        metadata = [
            {"placeholder": "[SCREENSHOT: https://a, A]", "heading": "## 章节 A"},
            {"placeholder": "[SCREENSHOT: https://b, B]", "heading": "## 章节 B"},
            {"placeholder": "[SCREENSHOT: https://c, C]", "heading": "## 章节 C"},
        ]
        points = _pick_refill_insert_points(draft, metadata, 2)
        self.assertEqual(points, [])

    def test_need_zero_returns_empty(self):
        draft = self._draft_with_headings()
        self.assertEqual(_pick_refill_insert_points(draft, None, 0), [])

    def test_no_headings_returns_empty(self):
        draft = "# 标题\n\n无章节正文。\n"
        self.assertEqual(_pick_refill_insert_points(draft, None, 3), [])

    def test_heading_none_placeholder_ignored(self):
        """首屏前占位符（heading=None）不应污染覆盖集合。"""
        draft = self._draft_with_headings()
        headings = list(__import__("re").finditer(r"(?m)^##\s+.+$", draft))
        metadata = [
            {"placeholder": "[SCREENSHOT: https://pre, 首屏前]", "heading": None},
        ]
        points = _pick_refill_insert_points(draft, metadata, 1)
        # heading=None 不覆盖任何 ## heading → 第一个 heading 作为插入点
        self.assertEqual(points, [headings[0].end()])

    def test_picks_first_n_uncovered_in_article_order(self):
        """多个未覆盖 heading 时按文章出现顺序选。"""
        draft = self._draft_with_headings()
        # 元数据：只覆盖 A（占位符仍在 draft 里）
        full_draft = (
            "## 章节 A\n\n"
            "A 正文。\n"
            "[SCREENSHOT: https://a, A]\n"
            "## 章节 B\n\n"
            "B 正文。\n"
            "## 章节 C\n\n"
            "C 正文。\n"
        )
        metadata = [
            {"placeholder": "[SCREENSHOT: https://a, A]", "heading": "## 章节 A"},
        ]
        points = _pick_refill_insert_points(full_draft, metadata, 2)
        # B、C 未覆盖，按顺序取前 2 个
        b_offset = full_draft.index("## 章节 B")
        c_offset = full_draft.index("## 章节 C")
        self.assertEqual(len(points), 2)
        self.assertGreater(points[0], b_offset)
        self.assertGreater(points[1], c_offset)


if __name__ == "__main__":
    unittest.main()
