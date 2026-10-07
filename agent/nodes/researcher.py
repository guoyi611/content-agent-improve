import json
import re

from langchain_core.messages import SystemMessage, HumanMessage
from agent.state import AgentState
from agent.tools.search import search
from agent.tools.page_fetch import fetch_readable_text
from agent.llm import get_llm


def _extract_keywords(state: AgentState) -> list[str]:
    """
    根据主题和文章规划，拆解出 2~4 个搜索关键词。
    重试时会参考 critic 反馈调整搜索方向。
    """
    retry_count = state.get("retry_count", 0)
    critic_feedback = state.get("critic_feedback", "")

    system = (
        "你是信息检索专家。根据文章主题和规划，拆解出 3 个最佳搜索关键词，用于搜索最新资讯。\n"
        "关键词要具体精准，覆盖不同维度（如：产品本身、竞品对比、行业影响）。\n"
        "直接返回 JSON 数组，不要包含任何其他内容。\n"
        '格式：["关键词1", "关键词2", "关键词3"]'
    )

    user_parts = [
        f"文章主题：{state['topic']}",
        f"目标平台：{state['platform']}",
    ]
    outline = state.get("outline", "")
    if outline:
        user_parts.append(f"文章规划：{outline}")
    if retry_count > 0 and critic_feedback:
        user_parts.append(f"上一轮 Critic 反馈（请针对性补充搜索）：{critic_feedback}")

    res = get_llm().invoke([
        SystemMessage(content=system),
        HumanMessage(content="\n".join(user_parts)),
    ])

    try:
        text = res.content.strip().replace("```json", "").replace("```", "")
        # 2026-09-18: 网关对 GPT-5.6 relay 的流式转换吃掉响应开头的 `["`，
        # 残缺输出（kw1", "kw2"]）会让关键词解析整体失败。补回缺失的开头。
        if text and not text.startswith(("[", "{")) and re.search(r'"\s*]', text):
            text = '["' + text
        # 提取 JSON 数组
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            keywords = json.loads(match.group())
            if isinstance(keywords, list) and keywords:
                return [str(k) for k in keywords]
    except (json.JSONDecodeError, ValueError):
        pass

    return [state["topic"]]


def _full_page_text(url: str) -> str:
    """Compatibility wrapper for callers that only need the extracted text."""
    return fetch_readable_text(url)[0]


def _material_from_result(result: dict) -> str:
    url = str(result.get("url", "")).strip()
    snippet = str(result.get("content", "")).strip()
    raw_content = str(result.get("raw_content", "") or "").strip()
    if len(raw_content) >= 200:
        evidence = raw_content[:18000]
    else:
        full_text, fetch_error = fetch_readable_text(url) if url else ("", "")
        if full_text:
            evidence = full_text[:18000]
        else:
            evidence = snippet
            if url and fetch_error:
                print(
                    f"  [Researcher] 正文抓取失败，保留搜索摘要："
                    f"{url} ({fetch_error})"
                )
    return (
        f"标题：{result.get('title', '无标题')}\n"
        f"内容：{evidence}\n"
        f"来源：{url}"
    )


def researcher_node(state: AgentState) -> dict:
    """
    1. 拆解搜索关键词（基于主题+规划+critic反馈）
    2. 对每个关键词执行搜索并抓取命中页面正文
    3. LLM 去重提炼成素材摘要
    """
    retry_count = state.get("retry_count", 0)
    critic_feedback = state.get("critic_feedback", "")

    if retry_count > 0 and critic_feedback:
        print(f"\n[Researcher] 根据 Critic 反馈补充搜索（第 {retry_count + 1} 轮）...")
        print(f"  Critic 建议：{critic_feedback[:80]}")
    else:
        print("\n[Researcher] 开始搜索素材...")

    # Step 0: 拆解搜索关键词（基于 planner 的 outline）
    keywords = _extract_keywords(state)
    print(f"  关键词：{keywords}")

    # 历史素材已在 pre_researcher 中检索，直接复用
    history_context = state.get("history_context", "")

    logs: list[str] = []
    new_materials: list[str] = []

    # Step 1: 逐个关键词搜索（基于 outline 的精准补充搜索）
    # max_results=12：截图模式要求 12 个不同 URL 候选、最终至少 5 张成功，
    # 每词只搜 8 条时素材池太小，writer 拿不到足够多的真实页面 URL。
    for keyword in keywords:
        print(f"  搜索：{keyword}")
        results = search(keyword, max_results=12)

        for r in results:
            new_materials.append(_material_from_result(r))

        logs.append(f"📰 \"{keyword}\" 找到 {len(results)} 条结果")

    # 合并素材：新搜索 + 预搜索/上轮素材
    old_materials: list[str] = state.get("raw_materials", [])
    raw_materials = new_materials + old_materials

    # 只有新搜索和前置搜索都为空时才终止。补充搜索可能因引擎风控
    # 返回 0 条，但 pre_researcher 已经提供了可核验素材；不能把这些素材
    # 丢掉，否则一次补充搜索抖动就会错误地 research_failed。
    if not raw_materials:
        print("  🛑 总素材为 0——终止本轮，不进入写作")
        return {
            "keywords": keywords,
            "context": "",
            "raw_materials": [],
            "research_failed": True,
            "log": state.get("log", [])
            + [f"🔍 补充搜索关键词：{'、'.join(keywords)}"]
            + logs
            + ["🛑 本轮没有任何可核验素材，已停止生成文章"],
        }

    print(f"  共收集 {len(raw_materials)} 条原始素材（新 {len(new_materials)} 条，前置 {len(old_materials)} 条），开始提炼...")

    # Step 3: LLM 整理提炼
    joined = "\n\n---\n\n".join(raw_materials)
    if len(joined) > 20000:
        joined = joined[:20000] + "\n\n[内容过长，已截断]"

    summary_res = get_llm().invoke([
        SystemMessage(content=(
            "你是信息整理助手。请对以下搜索结果进行整理：\n"
            "1. 合并重复信息（同一事实只保留信息最完整的版本）\n"
            "2. 保留所有具体数据（数字、百分比、价格、日期、版本号）\n"
            "3. 保留所有人名、公司名、产品名、技术术语\n"
            "4. 保留有价值的直接引语和关键表述\n"
            "5. 保留来源 URL（写作时可用于引用）\n"
            "6. 每条可核验事实必须紧跟来源名+完整原始 URL（写作器会据此核对锚点）："
            "禁止只写素材编号、域名或\"某报道\"。\n"
            "7. URL 必须逐字复制自输入的搜索结果，不得改写、补全、猜测或生成输入中不存在的 URL。\n"
            "8. 输出结构清晰的素材摘要，1000~1500字；即使压缩内容，也不得删除事实对应的来源 URL。\n"
            "宁可多保留信息，也不要过度压缩。直接输出摘要内容，不要加前缀说明。"
        )),
        HumanMessage(content=(
            f"文章主题：{state['topic']}\n\n"
            + (f"上一轮写作的修改建议：{critic_feedback}\n请特别补充相关内容。\n\n" if critic_feedback and retry_count > 0 else "")
            + (f"历史相关素材（来自素材库）：\n{history_context}\n\n" if history_context else "")
            + f"新搜索结果：\n{joined}"
        )),
    ])

    context = summary_res.content.strip()
    # 摘要模型可能为了满足字数上限删掉 URL，即使 prompt 已要求保留。
    # 将原始结果中的 URL 做确定性尾部索引，保证证据锚点与截图候选永不丢失。
    source_urls = []
    for material in raw_materials:
        source = re.search(r"^来源：(.+)$", material, re.MULTILINE)
        url = source.group(1).strip() if source else ""
        if url and url not in source_urls:
            source_urls.append(url)
    missing_urls = [url for url in source_urls if url not in context]
    if missing_urls:
        context += "\n\n【原始来源 URL（摘要模型不得删除）】\n" + "\n".join(
            f"- {url}" for url in missing_urls
        )
    print(f"  素材摘要完成（{len(context)}字，来源 URL {len(source_urls)} 条）")

    return {
        "keywords": keywords,
        "raw_materials": raw_materials,
        "context": context,
        "log": state.get("log", [])
            + [f"🔍 补充搜索关键词：{'、'.join(keywords)}"]
            + logs
            + ["✅ 素材整理完成"],
    }
