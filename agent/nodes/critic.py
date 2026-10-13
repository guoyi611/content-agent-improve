import json
import re
from langchain_core.messages import SystemMessage, HumanMessage
from agent.state import AgentState
from agent.llm import get_llm
from agent.prompts.templates import WRITING_PRINCIPLES
from agent.tools.critic_guard import (
    apply_caps, sanitize_retry_action, validate_violations,
    compute_image_alignment_score,
)


def _title_evidence_violation(topic: str, context: str, draft: str) -> str:
    """Return a hard-gate message when a factual title claim lacks evidence."""
    title_match = re.search(r"^#\s+(.+)$", draft.strip(), re.MULTILINE)
    if not title_match:
        return ""
    title = title_match.group(1)
    # Treat numeric/date/version tokens as claims. They must occur in the
    # evidence context, not merely in the user-supplied topic or outline.
    claims = re.findall(r"\b(?:v?\d+(?:\.\d+){0,3}%?|\d{4}[-年/.]\d{1,2}(?:[-月/.]\d{1,2})?)\b", title, re.I)
    missing = [claim for claim in claims if claim not in context]
    if missing:
        return f"标题事实缺少素材出处：{', '.join(missing)}"
    return ""


CRITIC_SYSTEM = """你是一位资深内容编辑，负责评估文章质量并给出改进建议。

## 重要前提
- 素材里真实出现的数据、版本号、日期，不要用你自己的知识去否定（你的知识可能过时）。
- 反过来，"素材里找不到、却出现在文章里"的具体数字/版本号/日期/标准号，按编造处理。
- 只在文章与素材明显矛盾、或文章内部前后数据不一致时才扣"事实纪律"以外的分。

## 评分标准（满分 9 分，由你打）
只评估文章的质量维度。**违规不在这里扣分** —— 违规写进 violations，由系统单独处理。
1. **信息密度**（3分）：内容是否充实，有没有空洞或注水
2. **可读性**（3分）：表达是否流畅、像人话，有没有吸引力；是否有可验证的具体数字/事件，而不是抽象判断
3. **结构完整性**（2分）：开头/正文/结尾是否完整，逻辑是否通顺；是否在结尾出现强行总结或鸡汤式升华
4. **平台适配度**（1分）：语气、字数、格式是否符合目标平台风格

## 第 5 项：图文一致性（1 分，不由你打分 —— 由系统按 image_placeholders 元数据确定性算）
你**不需要**为这一项打分，也不写入 score。但请在输出 JSON 里填一个
`image_alignment` 字段，把你**能够直接观察**的客观情况写进去（用于后续扩展 LLM judge；当前版本系统主要靠自己算）：
- `missing_role_count`: 缺 (role: ...) 自注的占位符个数（你看 draft 直接数即可）
- `hero_count`: role=hero 的占位符个数
- `visual_topic_mismatch`: 占位符紧邻段落讨论的主题与占位符视觉元素明显冲突的占位符 placeholder 列表（无法判断时空数组）
- `ungrounded_screenshots`: 截图所在段落没有「据 XX / 官方文档 / 官网」等语义引用短语的 SCREENSHOT placeholder 列表（无法判断时空数组）

## 违规判定（写入 violations，不要影响上面的 score）
逐条对照，**只列确实触发的**，quote 必须是逐字摘录的原文：

1. **语言纪律**：标题必须是中文；正文无彩色 emoji；无翻译腔句式（不是…而是…、这意味着、首先…其次…最后）；无大厂黑话（赋能/抓手/闭环/沉淀/对齐/赛道/链路/兜底）。
   中文标题判定口径：中文主语+中文谓语、整句以中文语法骨架为主即视为中文标题；
   括注式英文补充（如「ZCode（智谱 AI 编程工具）」）与文件名/路径/URL 片段（如 .git、README、workspace.json）不算破坏中文标题；整句英文标题或英文单词超过一半才算违规。
2. **事实纪律**：从文章里抽 5 个具体数字/版本号/日期/标准号/百分比，逐一到素材里找出处。素材里找不到的算编造。
3. **证据纪律**：
   - 标题里的任何事实都必须在素材里能找到原文出处
   - 正文每个具体数字/事件/版本号后应标注来源（来源名 + URL）
   - 截图的 URL 是否在素材里出现，段落讨论是否与截图直接对应
   - 是否出现素材外的"自补"事实/数字/人物

**以上三类的判定范围以上面列出的条目为准。** 不在条目里的表达——即便你觉得不够好、偏主观、或者"读起来有点像"——都不要写成违规。每条违规记录都会被自动核验，不属实的会被丢弃。

## 放行条件
- 总分 = 你打的 4 项分（满分 9） + 系统算的图文一致性 1 分（满分 1）= 满分 10。
- 总分达到 7.5 分（含 7.5 分）即视为通过。
- 语言纪律 / 事实纪律 / 证据纪律命中时，系统按各自上限封顶（5 / 4 / 6 分）。**你不需要自己算封顶，也不要因此调整 score。**

{writing_principles}

## 输出格式
严格输出以下 JSON，不要输出任何其他内容：
```json
{{"score": <1到9的整数>, "violations": [{{"rule": "<语言纪律|事实纪律|证据纪律>", "quote": "<触发规则的原文片段>"}}], "feedback": "<具体的修改建议，100字以内>", "action": "<rewrite|research>", "image_alignment": {{"missing_role_count": <int>, "hero_count": <int>, "visual_topic_mismatch": [<placeholder 字符串>], "ungrounded_screenshots": [<placeholder 字符串>]}}}}
```

violations 的填写要求：
- quote 必须**逐字摘自文章**，不得改写、概括或拼接
- 只列**确实触发上述规则条目**的违规。规则里没写到的表达，即便你认为不够好，也不要列进来
- 没有违规时填空数组 []，不要为了显得严格而凑数

action 的判定（决定重试时是否补搜素材）：
- rewrite：所有问题靠改写、删减或格式调整就能解决——表达纪律、删除无出处内容、补标注格式
- research：至少有一条需要补充新素材才能解决——缺少关键证据、数据或案例，素材里根本没有
- 拿不准时填 research（宁多搜一轮，不空写）"""

CRITIC_USER = """## 目标平台：{platform}

## 主题：{topic}

## 文章规划（Planner 产出）
{outline}

## 搜索素材（Writer 的输入——评估文章证据时的唯一事实来源）
{context}

{history_section}

## 文章初稿（需要评估的内容）
{draft}"""


def critic_node(state: AgentState) -> dict:
    """
    给 Writer 的初稿打分。
    提供搜索素材和历史素材作为上下文，确保评估基于充分信息。
    返回 score（1~10）和 feedback（修改建议）。
    """
    retry_count = state.get("retry_count", 0)
    print(f"\n[Critic] 评估初稿（第 {retry_count + 1} 次）...")

    title_violation = _title_evidence_violation(
        state.get("topic", ""), state.get("context", ""), state.get("draft", "")
    )
    if title_violation:
        # This is deterministic and must not depend on an LLM remembering the
        # title rule. Keep the score below the pass line even if the model
        # would otherwise rate the prose highly.
        print(f"  [Critic] 硬闸拦截：{title_violation}")
        return {
            "critic_score": 3,
            "critic_feedback": title_violation,
            "log": state.get("log", []) + [f"🛑 {title_violation}"],
        }

    # 构建历史素材部分（如果有）
    history_context = state.get("history_context", "")
    history_section = ""
    if history_context:
        history_section = f"## 历史素材（RAG 召回）\n{history_context}"

    system = CRITIC_SYSTEM.format(writing_principles=WRITING_PRINCIPLES)
    user = CRITIC_USER.format(
        platform=state["platform"],
        topic=state["topic"],
        outline=state.get("outline", "（无规划）"),
        context=state.get("context", ""),
        history_section=history_section,
        draft=state["draft"],
    )

    def _parse(raw: str) -> tuple:
        """剥围栏、解析 JSON。成功返回 (parsed_dict, raw)；失败抛出异常。"""
        # 剥 ```...``` 围栏
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if len(lines) >= 2 and lines[-1].strip().startswith("```"):
                text = "\n".join(lines[1:-1]).strip()
            elif len(lines) >= 1:
                text = "\n".join(lines[1:]).strip()

        # 2026-09-18: 本地网关对 GPT-5.6 relay 的流式转换会吃掉响应开头的 `{"`
        # （首 chunk 缺首字节，实测稳定复现：{"hello":"world"} → hello":"world"}）。
        # 把缺失的开头补回去。
        if text.startswith('score"'):
            text = '{"' + text

        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not json_match:
            raise ValueError("no JSON found")
        parsed = json.loads(json_match.group())

        # 引文 quote 限长 50 字，避免超长引文冲垮下游
        if "violations" in parsed and isinstance(parsed["violations"], list):
            parsed["violations"] = [
                {**v, "quote": str(v.get("quote", ""))[:50]}
                for v in parsed["violations"]
            ]
        return parsed, text

    # 解析失败最多重试一次
    try:
        parsed, raw_used = _parse(res.content.strip())
    except Exception:
        print(f"  [Critic] 首轮解析失败，重试...")
        try:
            res = get_llm().invoke([
                SystemMessage(content=system),
                HumanMessage(content=user),
            ])
            parsed, raw_used = _parse(res.content.strip())
        except Exception:
            parsed = None
            raw_used = ""

    if parsed is None:
        score = 3
        feedback = "评分输出无法解析，按不通过处理"
        action = "research"
        print(f"  [Critic] 原始输出前200字：{raw_used[:200]}")
    else:
        score = int(parsed.get("score", 3))
        feedback = str(parsed.get("feedback", ""))
        violations = parsed.get("violations") or []
        action = sanitize_retry_action(parsed.get("action"))

    # LLM 打 4 项分满分 9（信息密度 3 / 可读性 3 / 结构 2 / 平台适配度 1）。
    # 总分 = LLM 9 分 + 系统算的图文一致性 1 分 = 10 分制；pass 阈值 7.5。
    score = max(1, min(9, score))

    # 违规判定逐条校验，封顶由代码算——不依赖模型自查规则。
    # 实测 critic 会把"我的判断是""测试的是…还是…"这类正常表达判成语言纪律
    # 违规，且从不安自己的规则封上限（说封顶 5 分却给 7、8 分）。
    if not violations:
        print("  [Critic] 未报告任何违规")
    accepted, rejected = validate_violations(
        violations, state.get("draft", ""), state.get("context", "")
    )
    for item in rejected:
        print(
            f"  [Critic] 驳回误判（{item.get('why', '')}）："
            f"{item.get('quote', '')[:30]}"
        )
    capped, hit_rules = apply_caps(score, accepted)
    if hit_rules:
        print(f"  [Critic] 采信硬闸 {hit_rules} —— {score} → {capped}")
    score = capped

    # 图文一致性 1 分（narrative_design 改动 3）：按 image_placeholders + article_type
    # 确定性算，不依赖 LLM judge —— 老文章（缺元数据）默认给满分，避免误伤。
    alignment_score, alignment_log = compute_image_alignment_score(state)
    score = score + alignment_score
    if alignment_log:
        print(f"  [Critic] 图文一致性：{alignment_score}/1（{alignment_log}）")
    else:
        print(f"  [Critic] 图文一致性：{alignment_score}/1（无元数据，按满分计入）")

    print(f"  评分：{score}/10")
    if feedback:
        print(f"  建议：{feedback[:80]}...")
    print(f"  [Critic] 重试动作：{action}（{'跳过搜索直接重写' if action == 'rewrite' else '补充搜索后重写'}）")

    return {
        "critic_score": score,
        "critic_feedback": feedback,
        "retry_action": action,
        "log": state.get("log", []) + [
            f"📝 Critic 评分：{score}/10 —— {feedback[:50]}"
        ],
    }
