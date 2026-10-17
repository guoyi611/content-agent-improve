import re

from langchain_core.messages import SystemMessage, HumanMessage
from agent.state import AgentState
from agent.llm import get_llm
from agent.prompts.templates import get_direction_text

# 一手实测承诺词：选题命中任一，即视为向读者承诺"作者亲测过"。但本流水线
# 的素材只能来自公开搜索——writer 没有也不可能有一手实测记录。口径不降级时，
# critic 按标题字面承诺验收，反复要求"补实测/补样本"，而补搜永远搜不到
# "作者自己的实测"，重试烧完仍失败（实测 5→7→7 平台期）。
# 列表刻意收窄："上手/体验"这类弱承诺词不降级，避免误伤资料型选题，
# 漏掉的由 critic 正常反馈兜底。未来接入一手素材输入通道后，应改为
# "承诺存在且无一手素材"才降级。
FIRST_PARTY_CLAIM_RE = re.compile(r"实战|实测|亲测|踩坑|试用记|体验记")

# 降级口径声明：由代码确定性拼进 outline 开头，不依赖 planner LLM 自觉转述。
# writer 和 critic 都读 outline，声明到位后两端按同一口径写作与验收。
CALIBER_DECLARATION = """【口径声明（系统追加，效力高于选题原文措辞）】
选题原文带一手实测承诺，但本文素材全部来自公开搜索，作者没有也不可能有一手实测记录。规划与写作按资料型口径执行：
1. 标题与正文不得以第一人称声称做过实测/试用/踩坑（"我试了/我实测/我跑了"）；"实测"仅可用于指代他人的公开实测，且就近标注来源名和 URL
2. 切入角度降级为资料型：功能梳理、公开实测汇总、上手路径整理，不承诺作者亲测
3. 第一人称只用于表达判断（"我的判断是"），不用于叙述经历
4. 评审按本口径验收：不因缺少作者实测数据（样本、前后对比、核对值）扣分或要求补搜实测"""

PLANNER_SYSTEM = """你是一位资深内容策划，负责分析选题并规划文章框架。

## 你的任务
结合用户给出的文章主题和预搜索获取的实时资讯，产出一份清晰的文章规划，包括：
- **切入角度**：这篇文章要表达什么核心观点或叙事线（不是泛泛而谈）
- **目标读者**：谁会看这篇文章，他们最关心什么
- **核心要点**：2~4 个要覆盖的关键内容模块，每个说明需要什么素材支撑（数据、案例、对比等）
- **差异化**：这篇文章和普通报道/搬运相比，独特价值在哪

## 重要
- 充分利用提供的搜索结果中的实时信息来指导规划
- 确保规划基于真实的最新资讯，而非猜测

## 输出格式
直接输出规划内容（纯文本），不要加 JSON 包裹，不要加前缀说明。
控制在 200 字以内，简明扼要。"""

PLATFORM_STYLE = {
    "wechat": (
        "公众号推文：1500~2500字，标题关键词前置（50字内），> blockquote 开场点明核心观点，"
        "## 分节展开，善用表格对比和代码块，语气 informative + opinionated，结尾互动引导评论。"
    ),
    "xiaohongshu": (
        # 2026-10-17: 原文是"emoji 标题"，与 writer 模板的中文语言纪律
        # （禁彩色 emoji）冲突——planner 按旧风格写规划会把 writer 推向违规
        "小红书笔记：400~600字，口语化中文标题（25字内），开门见山说结论，"
        "3~5个干货要点，口语化表达，结尾带 #标签。"
    ),
    "zhihu": (
        "知乎文章：1000~2000字，标题有观点锋芒，> blockquote 亮观点，"
        "## 分节用论据支撑，理性克制有独立见解，善用表格和引用。"
    ),
}


def planner_node(state: AgentState) -> dict:
    """
    分析主题，规划文章框架和角度。
    输出 outline 供 researcher/writer/critic 参考。
    """
    print("\n[Planner] 分析选题，规划文章框架...")

    topic = state["topic"]
    needs_downgrade = bool(FIRST_PARTY_CLAIM_RE.search(topic))

    direction_text = get_direction_text(state.get("direction", "tech"))
    platform = state["platform"]
    style_hint = PLATFORM_STYLE.get(platform, "")

    # 构建用户消息，包含预搜索结果
    user_parts = [
        f"文章主题：{topic}",
        f"目标平台：{platform}",
        f"平台风格：{style_hint}",
        f"内容方向：{direction_text}",
    ]

    if needs_downgrade:
        user_parts.append(
            "\n## 口径约束（最高优先级，覆盖主题原文的措辞）\n"
            "选题带一手实测承诺，但素材只能来自公开搜索，作者不可能有亲测记录。"
            "切入角度必须降级为资料型：功能梳理、公开实测汇总或上手路径整理，"
            "不得规划任何作者亲测环节；他人实测只作为引用素材（标注来源）。"
        )

    # 注入预搜索素材，让规划有据可依
    raw_materials = state.get("raw_materials", [])
    if raw_materials:
        materials_text = "\n\n".join(raw_materials[:4])
        if len(materials_text) > 3000:
            materials_text = materials_text[:3000] + "\n[已截断]"
        user_parts.append(f"\n## 预搜索获取的实时资讯\n{materials_text}")

    history_context = state.get("history_context", "")
    if history_context:
        user_parts.append(f"\n## 历史相关素材\n{history_context[:1500]}")

    res = get_llm().invoke([
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(content="\n".join(user_parts)),
    ])

    outline = res.content.strip()
    if needs_downgrade:
        # 口径声明由代码拼接而非依赖 LLM 转述：critic 和 writer 的验收与
        # 写作口径必须确定性对齐，prompt 层面"大概率遵守"在这里不够
        outline = f"{CALIBER_DECLARATION}\n\n{outline}"
        print("  [Planner] 检测到一手实测承诺，已降级为资料型口径")
    print(f"  规划完成（{len(outline)}字）")

    return {
        "outline": outline,
        "log": state.get("log", []) + ["📋 文章规划完成"],
    }
