"""narrative_design 改动 4 —— image_fetcher 按 article_type 路由的单元测试。

consulting 文章不应配 AI 生图；若 writer 仍输出 IMAGE 占位符（走错 prompt
分支），image_fetcher 强制走 Unsplash 兜底并记录告警。creative + 缺
article_type 文章保持旧行为（按 IMAGE_PROVIDER 走 AI/Unsplash）。
"""

import unittest
from unittest.mock import patch


# IMAGE_PROVIDER 取值："prompt" / "unsplash" / "screenshot" / "mixed" / "openai" 等。
# 走 "ai" 分支需 IMAGE_PROVIDER 既不是 prompt 也不是 unsplash —— 用空字符串
# 即可触发默认 _detect_provider 路径（_get_image_provider 对未知值回退到 ai）。
_DEFAULT_AI_SOURCE = "ai"


def _state(article_type=None, draft=""):
    return {
        "topic": "test",
        "platform": "wechat",
        "draft": draft,
        "article_type": article_type,
        "image_style": None,
        "log": [],
        "screenshot_source_urls": [],
        "screenshot_attempted_urls": [],
    }


def _patched_fetcher(monkeypatch, ai_calls, unsplash_calls, source=_DEFAULT_AI_SOURCE):
    """替换 IMAGE_PROVIDER 解析与两种抓取函数，记录调用次数。"""
    monkeypatch.setattr("agent.nodes.image_fetcher._get_image_provider", lambda: source)
    monkeypatch.setattr(
        "agent.nodes.image_fetcher._fetch_ai_with_unsplash_fallback",
        lambda ph, kw, **_: (ph, f"![{kw}](ai.png)\n", f"ai {kw}"),
    )
    monkeypatch.setattr(
        "agent.nodes.image_fetcher._fetch_unsplash",
        lambda ph, kw, **_: (ph, f"![{kw}](uns.png)\n", f"unsplash {kw}"),
    )


class ConsultingRoutingTests(unittest.TestCase):
    """consulting 文章遇 IMAGE 占位符 → Unsplash 兜底 + 告警。"""

    def test_consulting_with_image_placeholder_routes_to_unsplash(self):
        from agent.nodes import image_fetcher

        draft = (
            "# 评测\n\n"
            "## 功能 A\n\n"
            "正文据官网。\n"
            "[IMAGE: A hero illustration]\n"
        )
        ai_calls = []
        uns_calls = []

        with patch("agent.nodes.image_fetcher._get_image_provider", return_value=_DEFAULT_AI_SOURCE), \
             patch(
                 "agent.nodes.image_fetcher._fetch_ai_with_unsplash_fallback",
                 side_effect=lambda ph, kw, **_: (ai_calls.append(kw) or (ph, "![ai](x.png)\n", "ai")),
             ), \
             patch(
                 "agent.nodes.image_fetcher._fetch_unsplash",
                 side_effect=lambda ph, kw, **_: (uns_calls.append(kw) or (ph, "![uns](x.png)\n", "uns")),
             ), \
             patch("agent.nodes.image_fetcher.get_config", return_value=""):
            result = image_fetcher.image_fetcher_node(_state("consulting", draft))

        self.assertEqual(ai_calls, [], "consulting 文章不应触发 AI 生图")
        self.assertEqual(len(uns_calls), 1, "consulting 文章应改走 Unsplash")
        # 告警写入 log
        joined_log = " ".join(result["log"])
        self.assertIn("咨询类文章", joined_log)
        self.assertIn("writer 走错分支", joined_log)
        # Unsplash 路径下不应再标注 "AI 生图"
        self.assertNotIn("AI 生图", joined_log)

    def test_creative_with_image_placeholder_keeps_ai(self):
        from agent.nodes import image_fetcher

        draft = (
            "# 随笔\n\n"
            "## 开篇\n\n"
            "一些叙事。\n"
            "[IMAGE: creative hero]\n"
        )
        ai_calls = []
        uns_calls = []

        with patch("agent.nodes.image_fetcher._get_image_provider", return_value=_DEFAULT_AI_SOURCE), \
             patch(
                 "agent.nodes.image_fetcher._fetch_ai_with_unsplash_fallback",
                 side_effect=lambda ph, kw, **_: (ai_calls.append(kw) or (ph, "![ai](x.png)\n", "ai")),
             ), \
             patch(
                 "agent.nodes.image_fetcher._fetch_unsplash",
                 side_effect=lambda ph, kw, **_: (uns_calls.append(kw) or (ph, "![uns](x.png)\n", "uns")),
             ), \
             patch("agent.nodes.image_fetcher.get_config", return_value=""):
            result = image_fetcher.image_fetcher_node(_state("creative", draft))

        self.assertEqual(len(ai_calls), 1, "creative 文章保持 AI 生图路径")
        self.assertEqual(uns_calls, [], "creative 文章不应被强制 Unsplash")
        # 不应出现 consulting 告警
        joined_log = " ".join(result["log"])
        self.assertNotIn("writer 走错分支", joined_log)

    def test_missing_article_type_keeps_default_provider(self):
        from agent.nodes import image_fetcher

        draft = (
            "# 标题\n\n"
            "## 章节\n\n"
            "正文。\n"
            "[IMAGE: legacy article]\n"
        )
        ai_calls = []
        uns_calls = []

        with patch("agent.nodes.image_fetcher._get_image_provider", return_value=_DEFAULT_AI_SOURCE), \
             patch(
                 "agent.nodes.image_fetcher._fetch_ai_with_unsplash_fallback",
                 side_effect=lambda ph, kw, **_: (ai_calls.append(kw) or (ph, "![ai](x.png)\n", "ai")),
             ), \
             patch(
                 "agent.nodes.image_fetcher._fetch_unsplash",
                 side_effect=lambda ph, kw, **_: (uns_calls.append(kw) or (ph, "![uns](x.png)\n", "uns")),
             ), \
             patch("agent.nodes.image_fetcher.get_config", return_value=""):
            image_fetcher.image_fetcher_node(_state(None, draft))

        self.assertEqual(len(ai_calls), 1, "缺 article_type 时走默认 AI 路径")
        self.assertEqual(uns_calls, [])

    def test_consulting_with_only_screenshots_no_warning(self):
        from agent.nodes import image_fetcher

        # consulting 文章里只有 SCREENSHOT 占位符 → 走截图，不应触发 IMAGE 告警
        draft = (
            "# 评测\n\n"
            "## 章节\n\n"
            "据官网。\n"
            "[SCREENSHOT: https://example.com/docs/a, 官方页]\n"
        )
        with patch("agent.nodes.image_fetcher._get_image_provider", return_value="screenshot"), \
             patch(
                 "agent.nodes.image_fetcher._fetch_screenshot",
                 return_value=(
                     "[SCREENSHOT: https://example.com/docs/a, 官方页]",
                     "![shot](data/images/screenshot_1.png)\n",
                     "📸 截图：https://example.com/docs/a",
                     1234,
                     "https://example.com/docs/a",
                 ),
             ), \
             patch("agent.nodes.image_fetcher.get_config", return_value=""):
            result = image_fetcher.image_fetcher_node(_state("consulting", draft))

        joined_log = " ".join(result["log"])
        self.assertNotIn("writer 走错分支", joined_log)


if __name__ == "__main__":
    unittest.main()
