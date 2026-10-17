import unittest

from agent.nodes.critic import _extract_score_json, _parse_critique_output


SCORE_JSON = '{"score": 7, "violations": [], "feedback": "ok", "action": "rewrite"}'


class ExtractScoreJsonTests(unittest.TestCase):
    def test_pure_json(self):
        self.assertEqual(_extract_score_json(SCORE_JSON)["score"], 7)

    def test_preamble_with_braces(self):
        # 模型先写叙述且叙述含花括号：贪婪正则会从第一个 { 匹配到最后一个 } 而炸，
        # 增量扫描应跳过非法起点找到评分对象
        text = "好的，评估{开始}如下：\n" + SCORE_JSON
        self.assertEqual(_extract_score_json(text)["score"], 7)

    def test_trailing_garbage_with_braces(self):
        text = SCORE_JSON + "\n以上就是评估，详见{附录}。"
        self.assertEqual(_extract_score_json(text)["score"], 7)

    def test_skips_non_score_object(self):
        text = '{"备注": "无关对象"} ' + SCORE_JSON
        self.assertEqual(_extract_score_json(text)["score"], 7)

    def test_no_json_returns_none(self):
        self.assertIsNone(_extract_score_json("模型拒绝输出，没有任何 JSON。"))

    def test_nested_braces_inside_json(self):
        # JSON 内部嵌套对象：raw_decode 必须解析完整对象而非中途截断
        text = '{"score": 6, "violations": [{"rule": "语言纪律", "meta": {"a": 1}}]}'
        parsed = _extract_score_json(text)
        self.assertEqual(parsed["violations"][0]["meta"], {"a": 1})


class ParseCritiqueOutputTests(unittest.TestCase):
    def test_strips_json_fence(self):
        raw = "```json\n" + SCORE_JSON + "\n```"
        parsed, _ = _parse_critique_output(raw)
        self.assertEqual(parsed["score"], 7)

    def test_restores_eaten_prefix(self):
        # 网关流式转换吃掉开头的 `{"`（实测复现过的中转站故障）
        eaten = SCORE_JSON[2:]  # 'score": 7, ...}'
        parsed, _ = _parse_critique_output(eaten)
        self.assertEqual(parsed["score"], 7)

    def test_truncates_long_quotes(self):
        raw = '{"score": 5, "violations": [{"rule": "语言纪律", "quote": "' + "长" * 100 + '"}]}'
        parsed, _ = _parse_critique_output(raw)
        self.assertEqual(len(parsed["violations"][0]["quote"]), 50)

    def test_keeps_non_dict_violation_items(self):
        # 模型偶尔把 violations 填成字符串列表：不能因此整次解析失败，
        # 原样保留交给 validate_violations 按格式不符驳回
        raw = '{"score": 5, "violations": ["格式错误的条目"]}'
        parsed, _ = _parse_critique_output(raw)
        self.assertEqual(parsed["violations"], ["格式错误的条目"])

    def test_unparseable_raises(self):
        with self.assertRaises(ValueError):
            _parse_critique_output("完全不是 JSON 的输出")


if __name__ == "__main__":
    unittest.main()
