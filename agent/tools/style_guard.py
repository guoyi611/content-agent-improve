"""翻译腔句式的检测与清洗。

writer 拿到的是一份"禁用句式"清单，而 critic 判的是"读起来像不像翻译腔"
这个语义概念——两者维度不同。清单永远追不上概念的外延：日志里 writer 每轮
都在吐出清单上没有的新变体（"缺的是"、"不需要是…但必须是…"），补一条它换一条。

所以这里改用 LLM 通读全文做语义级改写，让交付给 critic 的稿件和它的判据
处在同一维度上。正则保留下来只做**复检**：改写完再扫一遍，命中说明还有漏网的。
"""

import re

from langchain_core.messages import HumanMessage, SystemMessage

from agent.llm import get_llm


# 已知模式的快速检测。只用于改写后的复检，不再作为"要不要改写"的触发条件——
# 它覆盖不全，拿它当闸门会漏掉未列举的变体。
_PATTERNS = (
    # 不是 X，而是 Y ／ 不是 X，是 Y
    re.compile(r"不是[^，。；！？\n]{1,40}?[，,]?\s*(?:而是|是)"),
    # 不在 X，而在 Y ／ 不在于 X，在于 Y
    re.compile(r"不(?:在|在于)[^，。；！？\n]{1,40}?[，,]?\s*(?:而在|在于)"),
    # 与其说 X，不如说 Y
    re.compile(r"与其说[^，。；！？\n]{1,40}?[，,]?\s*不如说"),
    # 倒装：X，不是 Y ／ X，不在于 Y
    re.compile(r"[^，。；！？\n]{2,40}[，,]\s*不(?:是|在于)[^，。；！？\n]{2,40}"),
    # 翻面写法：X，而非 Y ／ X，并非 Y —— "不是A而是B"倒过来说，结构没变
    re.compile(r"[^，。；！？\n]{2,40}[，,]\s*(?:而非|并非)"),
)

REWRITE_SYSTEM = """你是一名中文编辑，负责消除文章里的翻译腔。

请通读收到的整篇文章，把读起来像翻译腔的地方改写成自然的中文表达。

要处理的是"用否定反衬肯定"的对比修辞，例如：
· 不是 A，而是 B ／ 不是 A，是 B
· 不在 A，而在 B ／ 不在于 A，在于 B
· 与其说 A，不如说 B
· 不需要 A，但必须 B
· A 缺的是 B
· 既不是 A，也不是 B，更不是 C

**判断依据是结构，不是具体用了哪些词。** 这一点最容易做错：
否定词的形式不固定——不是／并非／而非／不在于／缺的是／真正…的是，都是同一种结构。
下面这些只是换了词、结构没变，仍然算翻译腔——
· 企业缺的并不是更强的模型，缺的是能进现场的工程师
  （"不是"换成"并不是"，还是先否定再反衬）
· FDE是决策者，而非"高级实施"
  （"不是A而是B"翻个面写成"是B而非A"，还是同一个对比结构）
· 区别在于谁掌握决策权，跟技术栈无关
  （"不在于…在于"倒过来说，还是同一个对比结构）
必须把整个对比拆掉，改成单纯的直接陈述：
· 企业真正需要的是能进现场的工程师
· FDE是一线决策者
· 决定成败的是谁掌握决策权

**标题同样要改**，不要因为它是标题就保留这个句式。

同时改写：这意味着、当…的时候、诚然…但是…、连续的"被…所…"被动句。

反过来，**不要过度改写**。普通的否定句不是翻译腔，例如"这不是问题"、
"他不在这里"、"系统不支持该接口"，一个字都不要动。

硬性要求：
1. 只改表达方式。任何事实、数字、专有名词、引用来源一律不得增删改动。
2. 保持 markdown 结构：标题层级（#）、列表、加粗（**）、表格、
   图片占位符（[IMAGE: ...] / [SCREENSHOT: ...]）原样保留。
3. **段落数量和分行位置必须与原文完全一致**，不得合并或拆分段落。
4. 不是翻译腔的地方一个字都不要动。
5. 只输出改写后的完整文章。不要说明、不要前言后记、不要用代码块包裹。"""


def find_violations(text: str) -> list[tuple[int, str]]:
    """返回 [(行号, 该行内容)]，行号从 0 开始。用于改写后的复检。"""
    return [
        (i, line)
        for i, line in enumerate(text.split("\n"))
        if any(p.search(line) for p in _PATTERNS)
    ]


def _unwrap_fence(text: str) -> str:
    """模型偶尔会用 ``` 包住整篇，剥掉。"""
    stripped = (text or "").strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.split("\n")
    if len(lines) >= 2 and lines[-1].strip().startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _check_safe(original: str, candidate: str) -> tuple[bool, str]:
    """改写结果的行数与长度都必须贴近原文，否则视为模型跑偏。"""
    if not candidate.strip():
        return False, "结果为空"
    original_lines = original.split("\n")
    candidate_lines = candidate.split("\n")
    if len(original_lines) != len(candidate_lines):
        return False, f"行数不一致 {len(original_lines)}→{len(candidate_lines)}"
    ratio = len(candidate) / max(len(original), 1)
    if ratio < 0.75 or ratio > 1.25:
        return False, f"长度变化过大 {ratio:.0%}"
    return True, ""


def strip_translationese(draft: str) -> tuple[str, int]:
    """用 LLM 通读全文改写翻译腔句式。

    返回 ``(处理后的文本, 改动行数)``。任何一步失败都退回原文——这层是增益，
    不该成为生成链路的单点故障。
    """
    if not draft.strip():
        return draft, 0

    try:
        res = get_llm("writer").invoke(
            [
                SystemMessage(content=REWRITE_SYSTEM),
                HumanMessage(content=draft),
            ]
        )
        candidate = _unwrap_fence(res.content)
    except Exception as exc:  # noqa: BLE001 - 改写失败必须让原文继续走完流程
        print(f"  [StyleGuard] 改写失败，保留原文：{type(exc).__name__}")
        return draft, 0

    ok, reason = _check_safe(draft, candidate)
    if not ok:
        print(f"  [StyleGuard] 改写未通过校验（{reason}），保留原文")
        return draft, 0

    # 复检：已知模式仍能命中，说明这轮改写不干净。只提示，不否决——
    # 正则覆盖不全，拿它否决会误伤模型正确的语义判断。
    leftover = find_violations(candidate)
    if leftover:
        print(f"  [StyleGuard] 复检仍有 {len(leftover)} 行命中已知模式")

    changed = sum(
        1
        for a, b in zip(draft.split("\n"), candidate.split("\n"))
        if a != b
    )
    return candidate, changed
