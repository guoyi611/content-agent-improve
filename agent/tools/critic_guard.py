"""校验 critic 给出的违规判定，并由代码计算硬闸封顶。

critic 是 LLM，它对"翻译腔"这类概念会自行扩展判定范围。实测它把
"我的判断是"、"测试的是……还是……"、"究竟应该握在算法手里还是医生手里？"
这些正常表达都判成语言纪律违规，三轮重写全部卡死——而规则文本里根本没有
这些条目。

所以改为：critic 把违规写成结构化的 ``{"rule", "quote"}``，由这里逐条校验
——引文里**确实含有**该规则禁止的元素才算数，其余作废。封顶也由代码算，
不再依赖模型"记得按规则给你封上限"（它从来记不住）。

判定从严会有漏网之鱼（真违规但引文不典型 → 放过），这是有意的取舍：
误判会让好文章永远过不了，漏判只是少扣一次分。
"""

import re

from agent.tools.style_guard import TRANSLATIONESE_PATTERNS

# 彩色 emoji：U+1F000-U+1FAFF 覆盖绝大多数（🚀💡🔥🟢🔴…），
# U+2B00-U+2BFF 补上 ⭐ 一类的补充符号。
_COLOR_EMOJI = re.compile(r"[\U0001F000-\U0001FAFF⬀-⯿]")
# U+2600-U+27BF 这一段彩色符号与纯文本符号混在一起，只能逐个甄别：
# ✓✗⚠★ 是 ZH_DISCIPLINE 明确允许的，✅✨ 一类的彩色符号要判违规。
_MISC_SYMBOLS = re.compile(r"[☀-➿]")
_ALLOWED_TEXT_SYMBOLS = frozenset("✓✗⚠★")

# 语言纪律里逐字列出的其余禁用项
_FIXED_PHRASES = (
    re.compile(r"这意味着"),
    re.compile(r"当[^，。；！？\n]{1,20}的时候"),
    re.compile(r"诚然[^，。；！？\n]{1,30}但是"),
)

_BUZZWORDS = (
    "赋能", "抓手", "闭环", "沉淀", "对齐", "赛道", "链路",
    "心智", "势能", "兜底", "底层逻辑", "颗粒度",
)

# 各硬闸命中后的总分上限，与 CRITIC_SYSTEM 里写的一致
# 引号包裹的内容：技术引用（prompt injection 载荷、法规原文、受访者原话）
# 必须逐字保留，改了就失真。判定语言纪律时只看作者自己的行文。
_QUOTED = re.compile(
    r'"[^"]*"'                    # ASCII 双引号
    r"|“[^”]*”"    # “ ”
    r"|‘[^’]*’"    # ‘ ’
    r"|「[^」]*」"                 # 「 」
    r"|『[^』]*』"                 # 『 』
)

RULE_CAPS = {
    "语言纪律": 5,
    "事实纪律": 4,
    "证据纪律": 6,
}


def _strip_quoted(text: str) -> str:
    """剥掉引号内的内容，只留下作者自己的行文。"""
    return _QUOTED.sub("", text)


def _quotes_forbidden_element(quote: str) -> bool:
    """引文里（引号之外的部分）是否真的含有语言纪律禁止的元素。

    引号内豁免：那是在引用别人的原话或技术载荷，改了反而失真。
    """
    text = _strip_quoted(quote)
    if _COLOR_EMOJI.search(text):
        return True
    if any(ch not in _ALLOWED_TEXT_SYMBOLS for ch in _MISC_SYMBOLS.findall(text)):
        return True
    if any(p.search(text) for p in TRANSLATIONESE_PATTERNS):
        return True
    if any(p.search(text) for p in _FIXED_PHRASES):
        return True
    if any(word in text for word in _BUZZWORDS):
        return True
    return False


_NUMBER = re.compile(r"\d+(?:\.\d+)?")

# 来源标注的常见形态：据XX / 来源：/ URL
_SOURCE_HINT = re.compile(r"据[^，。；\n]{0,24}|来源[:：]|https?://")


def _fact_violation_is_real(quote: str, context: str) -> bool:
    """事实纪律是否成立：引文里的数字确实在素材中找不到。

    没有素材可对照时保守采信——那说明文章里的数字本就无从核实。
    """
    if not context:
        return True
    numbers = _NUMBER.findall(quote)
    if not numbers:
        return False  # 引文里没有数字，事实纪律无从谈起
    haystack = re.sub(r"[\s,]", "", context)
    return any(n not in haystack for n in numbers)


def _evidence_violation_is_real(quote: str) -> bool:
    """证据纪律是否成立：引文里的数字确实没有跟着来源标注。"""
    if not _NUMBER.findall(quote):
        return False
    return not _SOURCE_HINT.search(quote)


def validate_violations(
    violations: object, draft: str = "", context: str = ""
) -> tuple[list[dict], list[dict]]:
    """把 critic 给出的违规逐条校验，返回 ``(采信的, 驳回的)``。

    draft 用于确认引文确实出自原文，避免模型凭印象编造片段；
    context 是搜索素材，用于核对事实纪律的"编造"指控是否属实。
    """
    accepted: list[dict] = []
    rejected: list[dict] = []
    if not isinstance(violations, list):
        return accepted, rejected

    normalized_draft = re.sub(r"\s+", "", draft or "")

    for item in violations:
        if not isinstance(item, dict):
            continue
        rule = str(item.get("rule", "")).strip()
        quote = str(item.get("quote", "")).strip()

        if rule not in RULE_CAPS or not quote:
            rejected.append({"rule": rule, "quote": quote, "why": "缺少规则名或引文"})
            continue

        # 引文必须真实存在于正文中（忽略空白差异）
        if normalized_draft and re.sub(r"\s+", "", quote) not in normalized_draft:
            rejected.append({"rule": rule, "quote": quote, "why": "引文不在正文中"})
            continue

        # 语言纪律是误判重灾区，必须引文里真的含违禁元素
        if rule == "语言纪律" and not _quotes_forbidden_element(quote):
            rejected.append({"rule": rule, "quote": quote, "why": "引文不含违禁元素"})
            continue

        # 事实纪律：引文里的数字若在素材中查得到，就不是编造
        if rule == "事实纪律" and not _fact_violation_is_real(quote, context):
            rejected.append({"rule": rule, "quote": quote, "why": "引文数字在素材中可查"})
            continue

        # 证据纪律：引文本身若已带来源标注，就不能说它没标
        if rule == "证据纪律" and not _evidence_violation_is_real(quote):
            rejected.append({"rule": rule, "quote": quote, "why": "引文已含来源标注"})
            continue

        accepted.append({"rule": rule, "quote": quote})

    return accepted, rejected


def apply_caps(base_score: int, violations: list[dict]) -> tuple[int, list[str]]:
    """按采信的违规计算封顶后的总分，返回 ``(最终分, 命中的规则名)``。"""
    rules = sorted({str(v.get("rule", "")) for v in violations if v.get("rule")})
    caps = [RULE_CAPS[r] for r in rules if r in RULE_CAPS]
    if not caps:
        return base_score, []
    return min(base_score, min(caps)), rules
