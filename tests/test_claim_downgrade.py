import unittest
from unittest.mock import MagicMock, patch

from agent.nodes.critic import CRITIC_SYSTEM
from agent.nodes.planner import (
    CALIBER_DECLARATION,
    FIRST_PARTY_CLAIM_RE,
    planner_node,
)
from agent.prompts.templates import build_prompt


class FirstPartyClaimRegexTests(unittest.TestCase):
    """承诺词判定：命中即触发降级。列表刻意收窄（"上手/体验"这类弱词
    不降级），宁漏勿滥——漏掉的由 critic 正常反馈兜底。"""

    def test_hits_first_party_claims(self):
        for topic in (
            "【Skill 实战】OfficeCLI 批量排版 Word",
            "LangChain 实测报告",
            "亲测好用的三款翻译插件",
            "我的踩坑记：从 WordPress 迁移到 Hugo",
            "某工具试用记",
        ):
            with self.subTest(topic=topic):
                self.assertIsNotNone(FIRST_PARTY_CLAIM_RE.search(topic))

    def test_allows_resource_style_topics(self):
        for topic in (
            "OfficeCLI 上手指南：公开资料里能确认的能力边界",
            "2026 年开源编辑器横评",
            "关于 MCP，大多数人理解的都是错的",
        ):
            with self.subTest(topic=topic):
                self.assertIsNone(FIRST_PARTY_CLAIM_RE.search(topic))


def _run_planner(topic):
    """跑一遍 planner_node，返回 (state 更新, 发给 LLM 的 user prompt)。"""
    state = {
        "topic": topic,
        "platform": "zhihu",
        "direction": "tech",
        "raw_materials": [],
        "log": [],
    }
    with patch("agent.nodes.planner.get_llm") as mock_llm:
        res = MagicMock()
        res.content = "规划正文"
        mock_llm.return_value.invoke.return_value = res
        result = planner_node(state)
        prompt = mock_llm.return_value.invoke.call_args[0][0][1].content
    return result, prompt


class PlannerDowngradeTests(unittest.TestCase):
    def test_downgrades_first_party_topic(self):
        # 声明由代码拼接进 outline 开头（不依赖 LLM 转述），
        # 同时规划 prompt 里注入口径约束让 LLM 本身按降级角度规划
        result, prompt = _run_planner("【Skill 实战】OfficeCLI 批量排版")
        self.assertTrue(result["outline"].startswith(CALIBER_DECLARATION))
        self.assertIn("规划正文", result["outline"])
        self.assertIn("口径约束", prompt)

    def test_keeps_resource_topic_as_is(self):
        result, prompt = _run_planner("2026 年开源编辑器横评")
        self.assertEqual(result["outline"], "规划正文")
        self.assertNotIn("口径约束", prompt)


class XiaohongshuEvidenceTests(unittest.TestCase):
    """小红书模板与 critic 规则的对称性（2026-10-17 修复）：
    模板必须带上证据纪律，且不再教唆亲测口吻标题。"""

    def setUp(self):
        self.prompt = build_prompt("xiaohongshu", topic="T", context="C")

    def test_injects_evidence_discipline(self):
        self.assertIn("证据纪律", self.prompt)
        self.assertIn("来源名", self.prompt)

    def test_title_examples_no_longer_coach_fabricated_experience(self):
        self.assertNotIn("试了3天", self.prompt)


class CriticCaliberRuleTests(unittest.TestCase):
    def test_critic_knows_caliber_declaration(self):
        # 口径声明靠 outline 传递，critic 系统提示必须知道如何对待它
        self.assertIn("口径声明", CRITIC_SYSTEM)


if __name__ == "__main__":
    unittest.main()
