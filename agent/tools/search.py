from agent.config import get_config
import json
import time
import urllib.parse
import urllib.request

_tavily_client = None
_searxng_url: str | None | ValueError = False  # False = 未探测


def _get_tavily_client():
    global _tavily_client
    if _tavily_client is None:
        from tavily import TavilyClient
        api_key = get_config("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("未配置 TAVILY_API_KEY，请在设置中填写")
        _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client


def _get_searxng_url():
    """返回本地 searXNG JSON 端点；未配置/不可用返回 None。"""
    global _searxng_url
    if isinstance(_searxng_url, ValueError):
        raise _searxng_url
    if _searxng_url is False:
        raw = get_config("SEARXNG_URL")
        _searxng_url = raw if raw else None
    return _searxng_url


def _searxng_search(keyword: str, max_results: int) -> list[dict]:
    """本地 searXNG，返回与 Tavily 同构的 title/content/url 列表。

    指定中文引擎（搜狗微信）：默认引擎集合里 baidu/quark 常被
    CAPTCHA 挂起、sogou 网页会 crash、bing 对 CJK 查询匹配失灵
    （会混入英文无关结果且顺序不定）。sogou wechat 实测对中文科技
    新闻检索质量最高且稳定。
    """
    endpoint = _get_searxng_url().rstrip("/") + "/search"
    # 引擎序列：bing/google 优先——它们的 URL 是真实页面，截图模式可直接抓图；
    # sogou wechat 质量高但返回的是一次性跳转链（weixin.sogou.com/link?url=...），
    # 既会撞限流（挂死 20s+ 或软失败 0 条），又无法作为截图目标。仅作兜底。
    # 2026-09-18：bing/google/baidu/quark 全被 CAPTCHA/风控挂起，duckduckgo web
    # 与 privacywall 实测存活且质量高（ddg 直接命中 ai.meta.com/9to5mac 原文），
    # 提到最前；bing,google 与 sogou wechat 保留为后续兜底。
    # 2026-09-25：clash 7897 挂掉后 ddg/privacywall 直连=硬墙双超时（每组先交
    # 8~15s 超时税再由下一组救场），bing 直连实测稳定 10 条。首选组换回 bing；
    # ddg 组保留作兜底——clash 恢复后探活 15s 内自动复活，届时把它挪回首位。
    engine_sets = [
        "bing",
        "sogou wechat",
        "duckduckgo web,privacywall",
    ]
    last_err: Exception | None = None
    data: dict = {"results": []}
    for engines in engine_sets:
        query = urllib.parse.urlencode({
            "q": keyword,
            "format": "json",
            "language": "zh-CN",
            "engines": engines,
        })
        req = urllib.request.Request(endpoint + "?" + query, headers={
            "User-Agent": "Mozilla/5.0 (content-agent)",
        })
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.load(resp)
            if data.get("results"):
                break
            # HTTP 200 但 0 条 = sogou 被限流的软失败，同样降级
            print(f"  [Search] 引擎组 {engines!r} 返回 0 条，尝试下一组")
        except Exception as e:  # noqa: BLE001 — 任何网络错误都降级到下一组引擎
            last_err = e
            print(f"  [Search] 引擎组 {engines!r} 失败，尝试下一组：{type(e).__name__}")
    else:
        # 所有引擎组都"软失败"（HTTP 200 但 0 条）时 last_err 是 None，
        # 直接 raise 会抛出 "exceptions must derive from BaseException"。
        # 0 结果是空结果不是异常：返回空列表，让上层按"搜索无结果"处理。
        if last_err is not None:
            print(f"  [Search] 所有引擎组均失败，已放弃：{type(last_err).__name__}")
        else:
            print("  [Search] 所有引擎组均返回 0 条结果")
        return []
    results = []
    for item in data.get("results", [])[:max_results]:
        url = item.get("url", "")
        title = item.get("title", "")
        content = item.get("content", "")
        # sogou wechat 返回的是一次性跳转链接：截图会拍到搜狗中转页而非
        # 文章页，且链接触发限流。过滤掉，不让它进入素材。
        if "weixin.sogou.com" in url:
            continue
        results.append({"title": title, "content": content, "url": url})
    return results


def search(keyword: str, max_results: int = 5) -> list[dict]:
    """
    搜索关键词，返回结构化结果列表。
    每条结果包含 title、content、url 字段。
    优先 Tavily；未配置 Tavily key 时降级到本地 searXNG。
    """
    use_tavily = bool(get_config("TAVILY_API_KEY"))

    # Tavily requests can be reset by local/proxy network paths. Retry briefly,
    # then degrade to an empty result so the LLM can continue without live search.
    if use_tavily:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                search_kwargs = {
                    "max_results": max_results,
                    "include_raw_content": "markdown",
                }
                response = _get_tavily_client().search(keyword, **search_kwargs)
                return response.get("results", [])
            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
        print(f"  [Search] Tavily 暂时不可用，已跳过：{keyword} ({last_error})")
        return []

    try:
        return _searxng_search(keyword, max_results)
    except ValueError as exc:
        print(f"  [Search] {exc}")
        return []
    except Exception as exc:
        print(f"  [Search] searXNG 不可用，已跳过：{keyword} ({exc})")
        return []
