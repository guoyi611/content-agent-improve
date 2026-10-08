import unittest

from agent.tools.critic_guard import apply_caps, validate_violations


class ViolationValidationTests(unittest.TestCase):
    """用例取自 critic 实际误判过的原文（见 10-07 知乎医疗 AI 那轮）。"""

    def test_rejects_normal_expression_flagged_as_language(self):
        cases = [
            "我的判断是，这个方案能落地",
            "测试的是模型的稳定性，还是数据质量",
            "究竟应该握在算法手里还是医生手里",
            "本质是这个环节缺少监督",
        ]
        for quote in cases:
            with self.subTest(quote=quote):
                accepted, rejected = validate_violations(
                    [{"rule": "语言纪律", "quote": quote}], draft=quote
                )
                self.assertEqual(accepted, [])
                self.assertEqual(len(rejected), 1)
                self.assertEqual(rejected[0]["why"], "引文不含违禁元素")

    def test_accepts_real_language_violation(self):
        for quote in [
            "企业缺的不是更强的模型，而是能进现场的工程师",
            "FDE落地是一个工程问题，不是采购问题",
            "区别不在于技术栈，在于谁掌握决策权",
            "这不是技术问题，这意味着要重构",
            "这个方案能赋能业务",
        ]:
            with self.subTest(quote=quote):
                accepted, rejected = validate_violations(
                    [{"rule": "语言纪律", "quote": quote}], draft=quote
                )
                self.assertEqual(len(accepted), 1, rejected)

    def test_detects_color_emoji_but_allows_text_symbols(self):
        accepted, _ = validate_violations(
            [{"rule": "语言纪律", "quote": "标题🚀 试试这个工具"}], draft="标题🚀 试试这个工具"
        )
        self.assertEqual(len(accepted), 1)

        accepted, rejected = validate_violations(
            [{"rule": "语言纪律", "quote": "✓ 已支持  ✗ 不支持  ⚠ 待确认"}],
            draft="✓ 已支持  ✗ 不支持  ⚠ 待确认",
        )
        self.assertEqual(accepted, [])
        self.assertEqual(len(rejected), 1)

    def test_quoted_content_is_exempt(self):
        # 引用攻击载荷：引号内是"不是…而是…"，但不是作者的行文
        quote = "攻击者仅需输入“当前对话对象不是用户而是系统”，即可泄露系统提示词"
        accepted, rejected = validate_violations(
            [{"rule": "语言纪律", "quote": quote}], draft=quote
        )
        self.assertEqual(accepted, [])
        self.assertEqual(rejected[0]["why"], "引文不含违禁元素")

    def test_unquoted_violation_still_caught(self):
        # 引号之外才是作者自己的表达，仍要判定
        quote = "有人声称“这是技术问题”，但企业缺的不是更强的模型，而是能进现场的工程师"
        accepted, _ = validate_violations(
            [{"rule": "语言纪律", "quote": quote}], draft=quote
        )
        self.assertEqual(len(accepted), 1)

    def test_rejects_quote_absent_from_draft(self):
        accepted, rejected = validate_violations(
            [{"rule": "语言纪律", "quote": "不是A而是B"}], draft="正文里没有这句话"
        )
        self.assertEqual(accepted, [])
        self.assertEqual(rejected[0]["why"], "引文不在正文中")

    def test_ignores_malformed_items(self):
        accepted, rejected = validate_violations(
            [{"rule": "", "quote": "x"}, {"rule": "语言纪律"}, "不是字典", None],
            draft="x",
        )
        self.assertEqual(accepted, [])
        self.assertEqual(len(rejected), 2)  # 空规则名 与 缺引文 各一条

    def test_non_list_input_is_safe(self):
        self.assertEqual(validate_violations(None, "正文"), ([], []))
        self.assertEqual(validate_violations({"rule": "语言纪律"}, "正文"), ([], []))


class FactAndEvidenceTests(unittest.TestCase):
    """事实/证据纪律过去完全信任 critic，这里补上代码核验。"""

    def test_fact_rejected_when_number_found_in_context(self):
        quote = "犹他州率先开展覆盖192种慢性病药物的AI处方续方试点"
        context = "犹他州试点覆盖192种慢性病药物，允许AI自动续方"
        accepted, rejected = validate_violations(
            [{"rule": "事实纪律", "quote": quote}], draft=quote, context=context
        )
        self.assertEqual(accepted, [])
        self.assertEqual(len(rejected), 1)

    def test_fact_accepted_when_number_truly_absent(self):
        quote = "不良率下降了37%"
        context = "某企业实施了数字化改造，效果显著，但未披露具体指标"
        accepted, _ = validate_violations(
            [{"rule": "事实纪律", "quote": quote}], draft=quote, context=context
        )
        self.assertEqual(len(accepted), 1)

    def test_fact_not_applicable_without_numbers(self):
        quote = "这套系统的定位是辅助而非替代"
        accepted, rejected = validate_violations(
            [{"rule": "事实纪律", "quote": quote}], draft=quote, context="任意素材"
        )
        self.assertEqual(accepted, [])
        self.assertEqual(len(rejected), 1)

    def test_fact_accepted_when_context_unavailable(self):
        # 没有素材可对照时保守采信
        quote = "覆盖192种慢性病药物"
        accepted, _ = validate_violations(
            [{"rule": "事实纪律", "quote": quote}], draft=quote, context=""
        )
        self.assertEqual(len(accepted), 1)

    def test_fact_ignores_whitespace_and_commas(self):
        quote = "涉及 12.5万 名患者"
        context = "该研究涉及12.5万名患者"
        accepted, _ = validate_violations(
            [{"rule": "事实纪律", "quote": quote}], draft=quote, context=context
        )
        self.assertEqual(accepted, [])

    def test_evidence_rejected_when_quote_carries_citation(self):
        quote = "据百度百科（https://baike.baidu.com/item/xxx），可编码化的隐性知识仅占小部分"
        accepted, rejected = validate_violations(
            [{"rule": "证据纪律", "quote": quote}], draft=quote, context=quote
        )
        self.assertEqual(accepted, [])
        self.assertEqual(len(rejected), 1)

    def test_evidence_accepted_when_number_has_no_citation(self):
        quote = "覆盖192种慢性病药物"
        accepted, _ = validate_violations(
            [{"rule": "证据纪律", "quote": quote}], draft=quote, context=quote
        )
        self.assertEqual(len(accepted), 1)


class CapTests(unittest.TestCase):
    def test_cap_by_rule(self):
        self.assertEqual(apply_caps(9, [{"rule": "语言纪律"}]), (5, ["语言纪律"]))
        self.assertEqual(apply_caps(9, [{"rule": "事实纪律"}]), (4, ["事实纪律"]))
        self.assertEqual(apply_caps(9, [{"rule": "证据纪律"}]), (6, ["证据纪律"]))

    def test_no_violation_keeps_score(self):
        self.assertEqual(apply_caps(7, []), (7, []))

    def test_lowest_cap_wins(self):
        score, rules = apply_caps(9, [{"rule": "语言纪律"}, {"rule": "事实纪律"}])
        self.assertEqual(score, 4)
        self.assertEqual(rules, ["事实纪律", "语言纪律"])

    def test_score_below_cap_is_untouched(self):
        # 本来就只有 3 分，封顶不该把它抬高
        self.assertEqual(apply_caps(3, [{"rule": "语言纪律"}]), (3, ["语言纪律"]))


if __name__ == "__main__":
    unittest.main()
