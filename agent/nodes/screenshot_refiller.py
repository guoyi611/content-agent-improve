"""Targeted screenshot source replacement without rewriting the article body."""

import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage

from agent.llm import get_llm
from agent.nodes.image_fetcher import (
    SCREENSHOT_PATTERN,
    _embedded_screenshot_urls,
    canonicalize_screenshot_url,
)
from agent.state import AgentState
from agent.tools.screenshot import preflight_screenshot_url
from agent.tools.search import search


REPLACEMENT_PATTERN = re.compile(
    r"\[REPLACEMENT:\s*(\d+)\s*\|\s*(https?://[^\s|\]]+)\s*\|\s*([^\]]+)\]"
)


def _normalize_url(url: str) -> str:
    return canonicalize_screenshot_url(url)


def _is_usable_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc) and parsed.path.rstrip("/") != ""


def _topic_query(topic: str) -> str:
    """Reduce a long writing brief to a safe search-sized subject.

    bing 对长中文串查询会返回完全无关的结果（实测搜 WeatherNext 3 全是
    Apple 页面）。因此优先提取主题里的英文/品牌词（产品名、公司名），
    只拼一个短英文 query；提取不到再退回首行截断。
    """
    # 品牌词：连续的英文/数字词（含大小写混合的产品名如 WeatherNext、OpenMAIC）
    brand_tokens = re.findall(r"[A-Za-z][A-Za-z0-9.+-]{2,}", topic)
    # 过滤常见非品牌词
    stop = {"the", "and", "for", "with", "new", "official", "api", "demo",
            "com", "www", "http", "https", "html"}
    brands = [t for t in brand_tokens if t.lower() not in stop]
    if brands:
        # 取前 4 个品牌词（保持出现顺序），如 "WeatherNext 3 DeepMind"
        return " ".join(brands[:4])
    title = re.search(r'文章用这个标题[“"]([^”"]+)[”"]', topic)
    if title:
        return title.group(1)[:160]
    first_line = next((line.strip() for line in topic.splitlines() if line.strip()), topic)
    return re.sub(r"https?://\S+", "", first_line)[:160].strip()


def _discover_candidates(
    topic: str,
    targets: list[tuple[int, str]],
    used_normalized: set[str],
    raw_materials: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Search and preflight real pages instead of asking the LLM to invent URLs.

    2026-09-25: candidates whose URL appears verbatim in raw_materials are
    returned as priority candidates flagged "material" — evidence anchoring
    (scripts/rerun_strict.py) only accepts screenshots whose URL exists in
    raw_materials, so material-sourced candidates are the only ones that can
    survive the checker. Search-discovered candidates remain available as a
    fallback but rank behind material URLs.
    """
    subject = _topic_query(topic)
    queries: list[str] = [
        f"{subject} official documentation guides",
        f"{subject} official API reference examples",
        f"{subject} official GitHub README releases",
        f"{subject} 官方网站 功能介绍",
        f"{subject} 官方 文档 使用",
        f"{subject} github repo example",
    ]
    # 注意：不要把占位符描述拼进 query。描述来自 writer 对失败 URL 的注解
    # （如"Google 香港中文首页，页脚版权…"），拼进去会把搜索带偏到无关站点。
    raw: list[tuple[str, str]] = []
    material_items: list[tuple[str, str]] = []
    seen: set[str] = set(used_normalized)

    # 优先池：raw_materials 里出现过的 URL。证据锚定只认素材 URL，搜索
    # 候选再"相关"也无法通过 rerun_strict 的 raw_materials 校验，所以
    # 素材 URL 排第一梯队（预检通过后）。
    for line in (raw_materials or []):
        for match in re.finditer(r"https?://[^\s\"'<>)\]，。；）】」』》〉]+", line):
            url = match.group(0).rstrip(".,;")
            normalized = _normalize_url(url)
            if normalized in seen or not _is_usable_url(url):
                continue
            seen.add(normalized)
            title = line.split("\n", 1)[0].removeprefix("标题：").strip() or "素材页面"
            material_items.append((url, title))

    for query in queries[:6]:
        items = search(query, max_results=10)
        print(f"  [Refiller] 搜索 {query!r} → {len(items)} 条")
        for item in items:
            url = str(item.get("url", "")).strip()
            normalized = _normalize_url(url) if url else ""
            if not url or normalized in seen or not _is_usable_url(url):
                continue
            seen.add(normalized)
            raw.append((url, str(item.get("title", "")).strip()))

    if not raw and not material_items:
        print("  [Refiller] 候选池为空（素材无 URL 且搜索 0 条）")
        return []
    with ThreadPoolExecutor(max_workers=min(6, len(material_items + raw))) as pool:
        all_items = material_items + raw
        checks = list(pool.map(lambda item: preflight_screenshot_url(item[0]), all_items))
    valid_all = [item for item, valid in zip(all_items, checks) if valid]
    valid_material = [item for item in valid_all[:len(material_items)]]
    valid_items = valid_all[len(material_items):]
    print(
        f"  [Refiller] 预检通过 {len(valid_all)}/{len(all_items)} 个候选"
        f"（素材直取 {len(valid_material)}）"
    )

    subject_tokens = {
        token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9.-]{2,}", subject)
        if token.lower() not in {"api", "pro", "the", "and", "with", "for"}
    }

    def score(item: tuple[str, str]) -> int:
        url, title = item
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        searchable = f"{parsed.path} {title}".lower()
        value = 0
        if host.startswith(("docs.", "doc.", "developer.", "developers.", "platform.")):
            value += 3
        if host == "github.com" and len([part for part in parsed.path.split("/") if part]) >= 2:
            value += 3
        if any(part in parsed.path.lower() for part in ("/docs", "/guides", "/manual", "/reference", "/examples", "/releases")):
            value += 2
        if any(token in host for token in subject_tokens):
            value += 5
        elif host == "github.com" and any(token in parsed.path.lower() for token in subject_tokens):
            value += 4
        elif any(token in searchable for token in subject_tokens):
            value += 1
        if "official" in title.lower() or "官方" in title:
            value += 1
        return value

    # 硬闸：与主题品牌词毫无交集的候选（相关分 0）整批丢弃。bing 对小众
    # 主题经常混入完全无关的结果（实测 WeatherNext 查询混入德国 ADAC
    # 汽车过路费页，opencloak 查询混入 zhidao/淘宝页——后者会作为截图
    # 进入文章，然后在证据锚定处被 REJECT，浪费整轮 30 分钟）。
    relevant = [item for item in valid_items if score(item) > 1]
    if relevant:
        return sorted(valid_material, key=score, reverse=True) + sorted(
            relevant, key=score, reverse=True
        )[:30]
    if valid_material:
        # 搜索候选全被硬闸拦下时，素材 URL 池仍可用——它们天然与主题相关
        # （研究员搜出来的），不经过 bing 相关分这道有噪声的闸。
        return sorted(valid_material, key=score, reverse=True)[:30]
    # 全部无相关且素材池为空：宁可空手回（上层会走 fail 路径）也不再放
    # 无关候选进截图——错图进文章后被锚定校验整篇 REJECT，比缺图代价更大。
    print("  [Refiller] 搜索候选全部与主题无关，且素材无可用 URL——放弃本轮补图")
    return []


def _pick_refill_insert_points(
    draft: str,
    image_placeholders: list[dict] | None,
    need: int,
) -> list[int]:
    """改动 4 (narrative_design)：按 state 元数据找未覆盖 heading 补图。

    旧行为：补图时无脑塞前 N 个 ``##`` heading，图片全堆到文章开头 —— 1500 字
    公众号配 3 张图时三张全在第一屏。新行为：先看 ``image_placeholders`` 里
    已有占位符覆盖了哪些 heading，剩下的 heading 按出现顺序挑 ``need`` 个
    插入点；全部覆盖时返回空，调用方会把多余候选追加到文末。

    缺元数据（老文章 / planner 旧版本）→ 降级到前 N 个 heading 旧行为。
    已被 ``_drop_unanchored_screenshots`` 删除的占位符不算覆盖（按字符串
    是否仍存在于 draft 判断）。
    """
    if need <= 0:
        return []
    headings = list(re.finditer(r"(?m)^##\s+.+$", draft))
    if not headings:
        return []
    if not image_placeholders:
        # 降级：缺元数据 → 前 N 个 heading（保留旧行为，不破坏老文章）
        return [h.end() for h in headings[:need]]
    covered = {
        p["heading"] for p in image_placeholders
        if p.get("heading") and p.get("placeholder") in draft
    }
    points: list[int] = []
    for h in headings:
        if h.group(0).strip() in covered:
            continue
        points.append(h.end())
        if len(points) >= need:
            break
    return points


def _top_up_shortfall(state: AgentState, draft: str, retry_number: int) -> dict:
    """No failed placeholders left but captures still below the minimum.

    Append fresh [SCREENSHOT:] placeholders from discovered candidates so the
    next image_fetcher round has real work to do, instead of burning the
    remaining budget on no-op rounds. Candidates come from raw_materials URLs
    first (only they can pass evidence anchoring), search-pool second.
    """
    from agent.nodes.image_fetcher import min_screenshots_for

    min_count = min_screenshots_for(state.get("platform", ""))
    already = len(_embedded_screenshot_urls(draft))
    need = min_count - already
    if need <= 0:
        return {
            "screenshot_retry_count": retry_number,
            "log": state.get("log", []) + ["截图数量已达标，无需补图"],
        }

    used_urls = _embedded_screenshot_urls(draft)
    used_urls.update(state.get("screenshot_source_urls", []))
    used_urls.update(state.get("screenshot_attempted_urls", []))
    used_normalized = {_normalize_url(url) for url in used_urls}
    discovered = _discover_candidates(
        state["topic"],
        [],
        used_normalized,
        raw_materials=state.get("raw_materials") or [],
    )
    picks: list[tuple[str, str]] = []
    for url, title in discovered:
        normalized = _normalize_url(url)
        if normalized in used_normalized:
            continue
        used_normalized.add(normalized)
        picks.append((url, title or "官方功能页面"))
        if len(picks) >= need:
            break

    if not picks:
        return {
            "screenshot_retry_count": retry_number,
            "log": state.get("log", []) + [
                f"补图第 {retry_number} 轮：无失败占位符但仅 {already} 张成功，"
                "候选池已耗尽，无法继续补图"
            ],
        }

    # 插入位置：按 image_placeholders 元数据找未覆盖 ## heading（改动 4）。
    # 缺元数据时 helper 内部降级到前 N 个 heading（保留旧行为）。
    additions = [f"[SCREENSHOT: {url}, {title}]" for url, title in picks]
    insert_points = _pick_refill_insert_points(
        draft, state.get("image_placeholders"), len(additions)
    )
    refilled_draft = draft
    for insert_at, snippet in sorted(zip(insert_points, additions), reverse=True):
        refilled_draft = (
            refilled_draft[:insert_at] + "\n\n" + snippet + refilled_draft[insert_at:]
        )
    leftover = additions[len(insert_points):]
    if leftover:
        refilled_draft = refilled_draft.rstrip() + "\n\n" + "\n\n".join(leftover) + "\n"
    print(f"  [Refiller] 截图不足（{already}/{min_count}），追加 {len(picks)} 个素材候选占位符")
    return {
        "draft": refilled_draft,
        "screenshot_retry_count": retry_number,
        "screenshot_retry_note": "",
        "screenshot_attempted_urls": sorted(used_urls),
        "log": state.get("log", []) + [
            f"补图第 {retry_number} 轮：截图不足 {already}/{min_count}，"
            f"追加 {len(picks)} 个候选占位符"
        ],
    }


def screenshot_refiller_node(state: AgentState) -> dict:
    """Replace failed screenshot placeholders, or top up when captures fall short.

    2026-09-25 starvation fix: when every remaining placeholder already
    captured successfully but the total is still below the platform minimum,
    the old code found no slots to replace and burned the remaining refill
    budget on empty rounds (topic 64: 1 capture + 3 no-op rounds → anchored=0).
    Now zero-slots-with-shortfall injects fresh placeholders from discovered
    candidates instead.
    """
    draft = state["draft"]
    slots = list(SCREENSHOT_PATTERN.finditer(draft))
    retry_number = state.get("screenshot_retry_count", 0) + 1

    if not slots:
        return _top_up_shortfall(state, draft, retry_number)

    used_urls = _embedded_screenshot_urls(draft)
    used_urls.update(state.get("screenshot_source_urls", []))
    used_urls.update(state.get("screenshot_attempted_urls", []))
    used_normalized = {_normalize_url(url) for url in used_urls}
    failed_hosts = sorted({urlparse(url).netloc.lower() for url in used_urls if urlparse(url).netloc})
    targets: list[str] = []
    for index, match in enumerate(slots, start=1):
        failed_url = match.group(1).strip()
        description = (match.group(2) or "").strip()
        used_urls.add(failed_url)
        used_normalized.add(_normalize_url(failed_url))
        targets.append(f"{index}. 原 URL：{failed_url}\n   图片用途：{description}")

    discovered = _discover_candidates(
        state["topic"],
        [(index, (match.group(2) or "").strip()) for index, match in enumerate(slots, start=1)],
        used_normalized,
        raw_materials=state.get("raw_materials") or [],
    )
    discovered_text = "\n".join(
        f"- {url} | {title or '官方页面'}" for url, title in discovered
    ) or "（没有通过预检的候选）"

    prompt = f"""你是网页截图的来源补图助手。不要重写、总结或输出文章正文。
文章主题：{state['topic']}

以下截图位置因白屏、失效、验证页、登录页、限流或画面重复而失败。请仅为每个位置选择一个新的、可公开访问的官方功能子页、官方文档页、官方 GitHub 示例/文档页、产品演示页或结果页。

硬性规则：
- 保持每个位置的图片用途，URL 必须和已失败/已使用 URL 不同。
- 同一页面的查询参数或锚点变化仍视为同一页面，禁止用来凑数。
- 不要官网首页、产品入口、登录/注册/认证页；不要新闻、论坛、博客、公众号或其他博主文章。
- 优先展示具体功能、操作、设置、数据结果或实际效果。
- 避免 API JSON 地址、模型社区页面、聚合排行榜和经常触发验证/限流的页面；如果官方文档站不稳定，改用该项目官方 GitHub 仓库中对应的 README、examples 或 docs 子目录页面。
- 每个位置按优先级给出最多 3 个不同候选，系统会预检并选择第一个真实可访问的页面；不要重复使用同一域名下相同路径。
- 只能从下面这批已联网检索且 HTTP 预检通过的候选中选择，禁止自行编造或改写 URL：
{discovered_text}
- 只输出下列格式，每个位置一行，不得输出其他内容：
  [REPLACEMENT: 序号 | https://官方子页URL | 中文具体图注]
  同一序号可以连续输出最多 3 行备选。

待补位置：
{chr(10).join(targets)}

本轮已失败或已使用的主机（除非没有其他官方来源，否则请避开）：
{chr(10).join(failed_hosts)}

禁止再次使用的 URL：
{chr(10).join(sorted(used_urls))}
"""

    response = get_llm().invoke([HumanMessage(content=prompt)])
    selected: dict[int, tuple[str, str]] = {}
    allowed_normalized = {_normalize_url(url) for url, _ in discovered}
    for match in REPLACEMENT_PATTERN.finditer(response.content.strip()):
        index = int(match.group(1))
        url = match.group(2).strip()
        description = match.group(3).strip()
        normalized = _normalize_url(url)
        if index in selected:
            continue
        if (
            index < 1
            or index > len(slots)
            or normalized in used_normalized
            or normalized not in allowed_normalized
            or not _is_usable_url(url)
            or not preflight_screenshot_url(url)
        ):
            continue
        selected[index] = (url, description)
        used_normalized.add(normalized)

    # Deterministic fallback: fill omitted slots from the verified search pool
    # instead of sending invented URLs into another Playwright round.
    available = iter(discovered)
    for index, slot in enumerate(slots, start=1):
        if index in selected:
            continue
        for url, title in available:
            normalized = _normalize_url(url)
            if normalized in used_normalized:
                continue
            selected[index] = (
                url,
                title or (slot.group(2) or "具体官方功能页面").strip(),
            )
            used_normalized.add(normalized)
            break

    refilled_draft = draft
    for index, match in reversed(list(enumerate(slots, start=1))):
        replacement = selected.get(index)
        if not replacement:
            continue
        url, description = replacement
        placeholder = f"[SCREENSHOT: {url}, {description}]"
        refilled_draft = (
            refilled_draft[:match.start()] + placeholder + refilled_draft[match.end():]
        )

    return {
        "draft": refilled_draft,
        "screenshot_retry_count": retry_number,
        "screenshot_retry_note": "",
        "screenshot_attempted_urls": sorted(used_urls),
        "log": state.get("log", []) + [
            f"补图第 {retry_number} 轮：已替换 {len(selected)}/{len(slots)} 个失败截图位置"
        ],
    }
