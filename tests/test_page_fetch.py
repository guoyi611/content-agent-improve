import unittest
from unittest.mock import Mock, patch

from agent.tools.page_fetch import fetch_readable_text


class PageFetchTests(unittest.TestCase):
    @patch("agent.tools.page_fetch.trafilatura.extract", return_value="x" * 250)
    @patch("agent.tools.page_fetch.requests.get")
    def test_proxy_failure_falls_back_to_direct_page(self, get, _extract):
        failed = Mock()
        failed.raise_for_status.side_effect = RuntimeError("proxy rejected")
        success = Mock()
        success.text = "<html>article</html>"
        success.raise_for_status.return_value = None
        get.side_effect = [failed, success]

        text, error = fetch_readable_text("https://example.com/article")

        self.assertEqual(len(text), 250)
        self.assertEqual(error, "")
        self.assertEqual(get.call_count, 2)
        self.assertEqual(get.call_args_list[0].kwargs["timeout"], (5.0, 15.0))

    @patch("agent.tools.page_fetch.trafilatura.extract", return_value="")
    @patch("agent.tools.page_fetch.requests.get")
    def test_returns_diagnostic_when_all_targets_fail(self, get, _extract):
        response = Mock()
        response.raise_for_status.side_effect = RuntimeError("blocked")
        get.return_value = response

        text, error = fetch_readable_text("https://example.com/article")

        self.assertEqual(text, "")
        self.assertIn("proxy", error)
        self.assertIn("direct", error)

    def test_skips_non_web_url(self):
        text, error = fetch_readable_text("javascript:void(0)")
        self.assertEqual(text, "")
        self.assertIn("非网页 URL", error)


if __name__ == "__main__":
    unittest.main()
