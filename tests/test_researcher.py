import unittest
from unittest.mock import patch

from agent.nodes.researcher import _material_from_result


class ResearcherFallbackTests(unittest.TestCase):
    @patch(
        "agent.nodes.researcher.fetch_readable_text",
        return_value=("", "proxy: HTTPError, HTTP 403; direct: HTTPError, HTTP 403"),
    )
    def test_page_failure_keeps_search_snippet(self, _fetch):
        material = _material_from_result(
            {
                "title": "被拦截的页面",
                "content": "这是搜索引擎返回的可用摘要。",
                "url": "https://example.com/article",
            }
        )

        self.assertIn("这是搜索引擎返回的可用摘要。", material)
        self.assertIn("https://example.com/article", material)

    @patch("agent.nodes.researcher.fetch_readable_text")
    def test_tavily_raw_content_skips_page_fetch(self, fetch):
        material = _material_from_result(
            {
                "title": "Tavily 原文",
                "content": "搜索摘要",
                "raw_content": "正文" * 150,
                "url": "https://example.com/article",
            }
        )

        fetch.assert_not_called()
        self.assertIn("正文", material)
        self.assertNotIn("搜索摘要", material)


if __name__ == "__main__":
    unittest.main()
