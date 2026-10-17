import unittest

from agent.tools.style_guard import _check_safe, find_violations


class TranslationeseDetectionTests(unittest.TestCase):
    def test_detects_common_variants(self):
        cases = [
            "企业缺的不是更强的模型，而是能进现场的工程师",
            "FDE不是高级实施，是一个新岗位",
            "关键不在四段本身，而在执行的决策权",
            "与其说这是技术问题，不如说是组织问题",
            # 倒装与"不在于"变体：只写前向模式会漏掉这两类
            "FDE落地是一个工程问题，不是采购问题",
            "区别不在于技术栈，在于谁掌握决策权",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(len(find_violations(text)), 1)

    def test_detects_fixed_phrases(self):
        # 这三类是"判罚端认识、清洗端放过"的不对称缺口，必须由 find_violations 兜住
        cases = [
            "这意味着交付范式已经改变。",
            "当你想改流程的时候，先看权限。",
            "诚然成本高，但是值得。",
            "诚然，成本很高，但是值得。",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(len(find_violations(text)), 1)

    def test_plain_negation_is_not_flagged(self):
        cases = [
            "这不是问题。",
            "他并不是工程师。",
            "系统不支持该接口。",
        ]
        for text in cases:
            with self.subTest(text=text):
                self.assertEqual(find_violations(text), [])

    def test_does_not_cross_sentence_boundary(self):
        # "不是" 与 "而是" 分处两句时不算违规，否则会误伤正常行文。
        text = "这不是重点。而是我们忽略了另一个问题。"
        self.assertEqual(find_violations(text), [])

    def test_reports_line_index(self):
        text = "第一行正常\n企业缺的不是模型，而是工程师\n第三行正常"
        self.assertEqual(find_violations(text), [(1, "企业缺的不是模型，而是工程师")])


class SafetyCheckTests(unittest.TestCase):
    def test_accepts_faithful_rewrite(self):
        original = "第一行\n企业缺的不是模型，而是工程师\n第三行"
        candidate = "第一行\n企业真正缺的是工程师，模型并非瓶颈\n第三行"
        ok, reason = _check_safe(original, candidate)
        self.assertTrue(ok, reason)

    def test_accepts_small_line_drift(self):
        # 模型实测稳定差 1~2 行（多/少一个空行），漂移 ≤3 行应放行。
        # 用例构造需保持总长度贴近，否则会撞上长度校验而非行数校验。
        original = "\n".join(["甲乙丙丁" for _ in range(8)])
        candidate = "\n".join(["甲乙丙丁戊" for _ in range(6)])
        ok, reason = _check_safe(original, candidate)
        self.assertTrue(ok, reason)

    def test_rejects_line_count_change(self):
        original = "\n".join(["一行内容若干字" for _ in range(8)])
        candidate = "\n".join(["一行内容若干字" for _ in range(3)])
        ok, reason = _check_safe(original, candidate)
        self.assertFalse(ok)
        self.assertIn("行数", reason)

    def test_rejects_empty_result(self):
        ok, _ = _check_safe("原文内容", "   ")
        self.assertFalse(ok)

    def test_rejects_length_drift(self):
        ok, reason = _check_safe("这是一行足够长的原文内容用来撑起长度基准", "短")
        self.assertFalse(ok)
        self.assertIn("长度", reason)


if __name__ == "__main__":
    unittest.main()
