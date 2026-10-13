"""narrative_design 改动 3 —— 图文一致性 1 分评分单元测试。

``compute_image_alignment_score`` 是纯确定性函数（不调 LLM），由 critic_node
在违规封顶之后、文章总分 = LLM 4 项 (0~9) + alignment (0/0.5/1) = 0~10 中调用。
"""

import unittest

from agent.tools.critic_guard import compute_image_alignment_score


def _ph(kind, role, offset=0, caption=None, heading="## a"):
    return {
        "placeholder": "[X]",
        "kind": kind,
        "heading": heading,
        "paragraph_index": 0,
        "char_offset": offset,
        "role": role,
        "reason": "",
        "caption": caption,
    }


def _state(article_type=None, placeholders=None, draft=""):
    return {"article_type": article_type, "image_placeholders": placeholders, "draft": draft}


class ImageAlignmentBackwardsCompatTests(unittest.TestCase):
    """老文章（缺元数据）必须按满分计入 —— 避免新规则误伤历史。"""

    def test_empty_state(self):
        score, log = compute_image_alignment_score(_state())
        self.assertEqual(score, 1.0)
        self.assertEqual(log, "")

    def test_no_placeholders_no_article_type(self):
        score, log = compute_image_alignment_score(_state(article_type="consulting"))
        self.assertEqual(score, 1.0)
        self.assertEqual(log, "")


class ImageAlignmentRoleTests(unittest.TestCase):
    """role 自注缺 / hero 数量扣分。"""

    def test_perfect_one_hero(self):
        ph = [_ph("image", "hero", offset=10)]
        score, _ = compute_image_alignment_score(_state("creative", ph))
        self.assertEqual(score, 1.0)

    def test_two_heroes_penalty_half(self):
        ph = [
            _ph("image", "hero", offset=0),
            _ph("image", "hero", offset=50, heading="## b"),
        ]
        score, _ = compute_image_alignment_score(_state("creative", ph))
        self.assertEqual(score, 0.5)

    def test_zero_heroes_penalty_half(self):
        ph = [_ph("image", "evidence", offset=10)]
        score, _ = compute_image_alignment_score(_state("creative", ph))
        self.assertEqual(score, 0.5)

    def test_missing_role_one_keeps_full_when_hero_present(self):
        """0.2 轻微扣分不丢分（hero=1 单独验证）。"""
        ph = [
            _ph("image", "hero", offset=0),
            _ph("image", "unassigned", offset=50, heading="## b"),
        ]
        score, _ = compute_image_alignment_score(_state("creative", ph))
        self.assertEqual(score, 1.0)

    def test_missing_role_five_capped_at_one(self):
        """5 张缺 role 自注 = 1.0 penalty，但加上 hero=0 = 1.5 → cap 1.0 → 0 分。"""
        ph = [_ph("image", "unassigned", offset=i * 10, heading=f"## {i}") for i in range(5)]
        score, _ = compute_image_alignment_score(_state("creative", ph))
        self.assertEqual(score, 0.0)


class ImageAlignmentRoutingTests(unittest.TestCase):
    """A4 路由一致性：consulting 不应纯 AI 生图；creative 不应纯截图。"""

    def test_consulting_all_image_penalties(self):
        ph = [_ph("image", "hero", offset=10)]
        score, log = compute_image_alignment_score(_state("consulting", ph))
        self.assertEqual(score, 0.5)
        self.assertIn("A4", log)

    def test_creative_all_screenshot_penalties(self):
        ph = [_ph("screenshot", "hero", offset=10, caption="d")]
        score, log = compute_image_alignment_score(_state("creative", ph))
        self.assertEqual(score, 0.5)
        self.assertIn("A4", log)

    def test_consulting_mixed_no_routing_penalty(self):
        ph = [
            _ph("image", "hero", offset=0),
            _ph("screenshot", "evidence", offset=100, caption="d", heading="## b"),
        ]
        draft = "## b\n据 OpenAI 官方文档...\n[SCREENSHOT: https://x, d]"
        score, _ = compute_image_alignment_score(_state("consulting", ph, draft))
        self.assertEqual(score, 1.0)

    def test_creative_mixed_no_routing_penalty(self):
        ph = [
            _ph("image", "hero", offset=0),
            _ph("screenshot", "evidence", offset=100, caption="d", heading="## b"),
        ]
        draft = "## b\n据 OpenAI 官方文档...\n[SCREENSHOT: https://x, d]"
        score, _ = compute_image_alignment_score(_state("creative", ph, draft))
        self.assertEqual(score, 1.0)

    def test_no_article_type_means_no_routing_check(self):
        ph = [_ph("image", "evidence", offset=10)]
        score, log = compute_image_alignment_score(_state(None, ph))
        # hero=0 → 0.5；但 A4 不应触发
        self.assertEqual(score, 0.5)
        self.assertNotIn("A4", log)


class ImageAlignmentSemanticCitationTests(unittest.TestCase):
    """引用截图所在段落必须含语义引用短语。"""

    def test_screenshot_with_source_hint_keeps_full(self):
        ph = [_ph("screenshot", "hero", offset=5)]
        draft = "## a\n据 OpenAI 官方文档（https://x）...\n[SCREENSHOT: https://x, d]"
        score, _ = compute_image_alignment_score(_state("consulting", ph, draft))
        self.assertEqual(score, 1.0)

    def test_screenshot_without_source_hint(self):
        """单张缺语义引用 → 0.2 扣分 → 仍 1.0（hero=1 OK）。"""
        ph = [_ph("screenshot", "hero", offset=5)]
        draft = "## a\n纯描述，没说据什么\n[SCREENSHOT: https://x, d]"
        score, _ = compute_image_alignment_score(_state("consulting", ph, draft))
        self.assertEqual(score, 1.0)

    def test_multiple_ungrounded_accumulate(self):
        """3 张截图分散在 3 个 heading，每段都无语义引用 → 累计 -0.6 + hero≠1 -0.5 = 1.1 → cap 1.0 → 0 分。"""
        ph = [
            _ph("screenshot", "hero", offset=5, heading="## a"),
            _ph("screenshot", "evidence", offset=5, heading="## b"),
            _ph("screenshot", "transition", offset=5, heading="## c"),
        ]
        draft = "## a\n[SCREENSHOT: x]\n纯描述\n## b\n[SCREENSHOT: y]\n纯描述\n## c\n[SCREENSHOT: z]\n纯描述"
        score, log = compute_image_alignment_score(_state("consulting", ph, draft))
        self.assertEqual(score, 0.0)


class ImageAlignmentStackingTests(unittest.TestCase):
    """多项扣分叠加：测试 cap 与最终 0 分路径。"""

    def test_stacked_penalties_cap_to_one(self):
        """3 项扣分各自 -0.5 / -0.5 / -0.5 → cap 1.0 → 0 分。"""
        ph = [
            _ph("image", "hero", offset=0),
            _ph("image", "hero", offset=50, heading="## b"),
        ]
        # hero=2 (-0.5) + missing role 0 + A4 consulting 全 AI (-0.5) = 1.0
        score, log = compute_image_alignment_score(_state("consulting", ph))
        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()