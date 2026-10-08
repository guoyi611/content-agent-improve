import json
import re
from langchain_core.messages import SystemMessage, HumanMessage
from agent.state import AgentState
from agent.llm import get_llm
from agent.prompts.templates import WRITING_PRINCIPLES
from agent.tools.critic_guard import apply_caps, validate_violations


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

## 评分标准（满分 10 分）
只评估文章的质量维度。**违规不在这里扣分** —— 违规写进 violations，由系统单独处理。
1. **信息密度**（3分）：内容是否充实，有没有空洞或注水
2. **可读性**（3分）：表达是否流畅、像人话，有没有吸引力；是否有可验证的具体数字/事件，而不是抽象判断
3. **结构完整性**（2分）：开头/正文/结尾是否完整，逻辑是否通顺；是否在结尾出现强行总结或鸡汤式升华
4. **平台适配度**（2分）：语气、字数、格式是否符合目标平台风格

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
- 总分达到 8 分（含 8 分）即视为通过。
- 语言纪律 / 事实纪律 / 证据纪律命中时，系统按各自上限封顶（5 / 4 / 6 分）。**你不需要自己算封顶，也不要因此调整 score。**

{writing_principles}

## 输出格式
严格输出以下 JSON，不要输出任何其他内容：
```json
{{"score": <1到10的整数>, "violations": [{{"rule": "<语言纪律|事实纪律|证据纪律>", "quote": "<触发规则的原文片段>"}}], "feedback": "<具体的修改建议，100字以内>"}}
```

violations 的填写要求：
- quote 必须**逐字摘自文章**，不得改写、概括或拼接
- 只列**确实触发上述规则条目**的违规。规则里没写到的表达，即便你认为不够好，也不要列进来
- 没有违规时填空数组 []，不要为了显得严格而凑数"""

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

    res = get_llm().invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    raw = res.content.strip()

    # 2026-09-18: 本地网关对 GPT-5.6 relay 的流式转换会吃掉响应开头的 `{"`
    # （首 chunk 缺首字节，实测稳定复现：{"hello":"world"} → hello":"world"}）。
    # critic 走流式 invoke，裸 {"score": ...} 必然被打残成 score": ...}。
    # 把缺失的开头补回去，避免评分输出被误判为无法解析。
    if raw.startswith('score"'):
        raw = '{"' + raw

    # 解析 JSON — 提取最外层 {}
    # 注意：解析失败不能默认给 7 分（通过）——那会让质量闸形同虚设。
    # 评不出来按"未通过"处理（3 分），并把原始输出打出来便于诊断。
    violations: list = []
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            score = int(parsed.get("score", 3))
            feedback = str(parsed.get("feedback", ""))
            violations = parsed.get("violations") or []
        except (json.JSONDecodeError, ValueError, TypeError):
            score = 3
            feedback = "评分输出无法解析，按不通过处理"
            print(f"  [Critic] 原始输出前200字：{raw[:200]}")
    else:
        score = 3
        feedback = "评分输出无法解析，按不通过处理"
        print(f"  [Critic] 原始输出前200字：{raw[:200]}")

    score = max(1, min(10, score))

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

    print(f"  评分：{score}/10")
    if feedback:
        print(f"  建议：{feedback[:80]}...")

    return {
        "critic_score": score,
        "critic_feedback": feedback,
        "log": state.get("log", []) + [
            f"📝 Critic 评分：{score}/10 —— {feedback[:50]}"
        ],
    }
