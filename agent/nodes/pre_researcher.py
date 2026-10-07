"""
前置搜索节点 —— 在 Planner 之前执行，提供实时信息支撑规划。

职责：
  1. 基于原始 topic 执行搜索，获取实时资讯
  2. 从向量库检索历史素材（RAG）
  3. 将结果存入 state，供 Planner 参考
"""

from agent.state import AgentState
from agent.tools.search import search
from agent.tools.page_fetch import fetch_readable_text
from agent.memory import search_similar
import re


def _search_query(topic: str) -> str:
    """Turn a long writing brief into a focused Tavily query."""
    title = re.search(r'文章用这个标题[“\"]([^”\"]+)[”\"]', topic)
    if title:
        return title.group(1)
    urls = re.findall(r'https?://[^\s"”]+', topic)
    head = re.split(r'[。\n]', topic, maxsplit=1)[0].strip()
    return " ".join(([head[:180]] if head else []) + urls[:1])


def _full_page_text(url: str) -> str:
    """Compatibility wrapper for callers that only need the extracted text."""
    return fetch_readable_text(url)[0]


def pre_researcher_node(state: AgentState) -> dict:
    """基于主题做初步搜索 + RAG 检索，为 Planner 提供实时信息。"""
    topic = state["topic"]
    query = _search_query(topic)
    print(f"\n[Pre-Researcher] 预搜索：{query}")

    logs: list[str] = []
    raw_materials: list[str] = []
    raw_content_count = 0
    page_content_count = 0
    page_failure_count = 0

    # 1. 搜索实时资讯（max_results=8，加正文抓取）
    results = search(query, max_results=8)
    for r in results:
        url = str(r.get("url", "")).strip()
        raw_content = str(r.get("raw_content", "") or "").strip()
        if len(raw_content) >= 200:
            content = raw_content[:18000]
            raw_content_count += 1
        else:
            full_text, fetch_error = fetch_readable_text(url) if url else ("", "")
            if full_text:
                content = full_text[:18000]
                page_content_count += 1
            else:
                content = str(r.get("content", "") or "").strip()
                if url and fetch_error:
                    page_failure_count += 1
                    print(
                        f"  [Pre-Researcher] 正文抓取失败，使用搜索摘要："
                        f"{url} ({fetch_error})"
                    )
        raw_materials.append(
            f"标题：{r.get('title', '无标题')}\n"
            f"内容：{content}\n"
            f"来源：{url}"
        )
    logs.append(
        f"📰 预搜索 \"{query}\" 找到 {len(results)} 条结果"
        f"（搜索原文 {raw_content_count}，页面正文 {page_content_count}，"
        f"抓取失败 {page_failure_count}）"
    )
    print(f"  搜索到 {len(results)} 条结果")

    # 2. 从向量库检索历史素材
    history_items = search_similar(query, k=3)
    history_context = ""
    if history_items:
        history_context = "\n\n---\n\n".join(history_items)
        logs.append(f"📚 从素材库找到 {len(history_items)} 条历史素材")
        print(f"  📚 从素材库找到 {len(history_items)} 条历史素材")

    print(f"  预搜索完成，收集 {len(raw_materials)} 条素材")

    return {
        "raw_materials": raw_materials,
        "history_context": history_context,
        "log": state.get("log", []) + logs,
    }
