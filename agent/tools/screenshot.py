"""
Playwright 网页截图工具 — 截取指定 URL 的可视区域作为文章插图

用法：
    from agent.tools.screenshot import take_screenshot
    path = take_screenshot("https://example.com", description="Example homepage")
    # → "data/images/screenshot_1711800000.png"

依赖：
    uv pip install playwright
    python -m playwright install chromium
"""

import base64
import json
import os
import time
import threading
from dataclasses import dataclass
from functools import wraps
from urllib.parse import quote, urlparse

import requests

from agent.config import API_BASE_URL


BLOCKED_PAGE_MARKERS = (
    "page not found",
    "404 not found",
    "404 page not found",
    "页面不存在",
    "页面未找到",
    "找不到该页面",
    "performing security verification",
    "verify you are human",
    "verifying you are human",
    "checking your browser",
    "just a moment",
    "cloudflare",
    "access denied",
    "captcha",
    "人机验证",
    "安全验证",
    "访问被拒绝",
)
RAW_HTML_CHALLENGE_MARKERS = (
    "cf-turnstile-response",
    "challenge-platform/h/g/turnstile",
    "hcaptcha-response",
    "g-recaptcha-response",
)
INCOMPLETE_PAGE_MARKERS = (
    "文档正在编辑中", "敬请等待", "敬请期待", "内容建设中",
    "暂无内容", "coming soon", "under construction",
)
THIRD_PARTY_CONTENT_HOSTS = (
    "mp.weixin.qq.com", "weixin.qq.com", "zhihu.com", "sohu.com", "163.com",
    "ifeng.com", "eastmoney.com", "sina.com.cn", "toutiao.com", "bilibili.com",
    "xiaohongshu.com", "instagram.com", "woshipm.com", "juejin.cn", "csdn.net",
    "medium.com", "36kr.com", "huxiu.com", "thepaper.cn", "reddit.com",
    "stackoverflow.com", "youtube.com", "youtu.be", "quora.com",
)
FORUM_HOST_PREFIXES = ("community.", "forum.", "forums.")
ENTRY_PATH_MARKERS = ("/login", "/signin", "/sign-in", "/register", "/signup", "/sign-up", "/auth")
# Reject explicit data-file URLs. Paths containing /api/ may still be real
# documentation pages (for example /api/docs/guides), so content-type
# preflight—not the path name—decides whether they are JSON endpoints.
API_PAGE_MARKERS = (".json", ".xml")
SCREENSHOT_ENGINE_VERSION = "screenshot-v27-search-verified-universal-20260819"
_PROBE_SESSION = requests.Session()
_PROBE_SESSION.headers.update(
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/131.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml",
    }
)
_PREFLIGHT_CACHE_LOCK = threading.Lock()


def _ttl_preflight_cache(func):
    """Cache stable successes longer than transient failures."""
    cache: dict[str, tuple[float, bool]] = {}

    @wraps(func)
    def wrapped(target_url: str) -> bool:
        now = time.monotonic()
        with _PREFLIGHT_CACHE_LOCK:
            cached = cache.get(target_url)
            if cached:
                created_at, value = cached
                ttl = 1800 if value else 60
                if now - created_at < ttl:
                    return value
        value = func(target_url)
        with _PREFLIGHT_CACHE_LOCK:
            cache[target_url] = (time.monotonic(), value)
        return value

    def cache_clear() -> None:
        with _PREFLIGHT_CACHE_LOCK:
            cache.clear()

    wrapped.cache_clear = cache_clear
    return wrapped
LANDING_ACCOUNT_MARKERS = ("登录", "注册", "sign in", "log in", "sign up")
LANDING_CTA_MARKERS = (
    "免费试用", "立即体验", "立即开始", "立即下载", "下载客户端",
    "下载 workbuddy", "get started", "try for free", "download",
)


def _is_blank_screenshot(filename: str) -> bool:
    """Reject all-white and nearly uniform browser canvases after capture."""
    try:
        from PIL import Image, ImageStat

        with Image.open(filename) as image:
            thumbnail = image.convert("RGB").resize((64, 40))
            stat = ImageStat.Stat(thumbnail)
            mean = sum(stat.mean) / 3
            deviation = sum(stat.stddev) / 3
            pixels = list(thumbnail.get_flattened_data())
        near_white_ratio = sum(
            1 for red, green, blue in pixels
            if red >= 245 and green >= 245 and blue >= 245
        ) / len(pixels)
        return (mean >= 248 and near_white_ratio >= 0.97) or deviation <= 1.5
    except (ImportError, OSError, ValueError, ZeroDivisionError):
        return False


def _is_entry_or_third_party_url(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower().split(":")[0]
    path = parsed.path.rstrip("/").lower()
    if any(host == domain or host.endswith(f".{domain}") for domain in THIRD_PARTY_CONTENT_HOSTS):
        return True
    if host.startswith(FORUM_HOST_PREFIXES):
        return True
    if any(marker in path for marker in ENTRY_PATH_MARKERS):
        return True
    if any(marker in path for marker in API_PAGE_MARKERS):
        return True
    # Root-URL rule intentionally removed: many official product homepages
    # ARE the canonical feature page (e.g. kimi.com is itself the AI chat
    # surface, notion.ai is the marketing/feature index). They were being
    # blanket-rejected here which starved the writer/refiller pipeline of
    # official screenshots. Visual/landing-page heuristics in the caller
    # still filter truly blank/marketing-only shells.
    return False


def _is_generic_landing_page(url: str, visible_text: str) -> bool:
    """Identify product entry/marketing pages that merely have a non-root slug."""
    path_segments = [segment for segment in urlparse(url).path.split("/") if segment]
    if len(path_segments) > 1:
        return False
    has_account_action = any(marker in visible_text for marker in LANDING_ACCOUNT_MARKERS)
    has_conversion_action = any(marker in visible_text for marker in LANDING_CTA_MARKERS)
    return has_account_action and has_conversion_action


@_ttl_preflight_cache
def preflight_screenshot_url(target_url: str) -> bool:
    """Cheaply reject dead/API/challenge URLs before launching Playwright.

    This is deliberately only a candidate filter; Playwright still performs
    the final visual/content validation. A GET fallback is used because many
    documentation CDNs do not implement HEAD correctly.
    """
    if _is_entry_or_third_party_url(target_url):
        return False
    try:
        # When SCREENSHOT_PROXY is configured, probe through the same reverse
        # proxy used by Chromium so the preflight reflects the same network
        # path that the screenshot capture will actually use. Without this,
        # domestic IP fetches time out for any host blocked behind GFW and the
        # screenshot candidate is wrongly rejected.
        proxy_prefix = os.environ.get("SCREENSHOT_PROXY", "").strip().rstrip("/")
        if proxy_prefix:
            api_key = os.environ.get("SCREENSHOT_PROXY_KEY", "").strip()
            headers = {"User-Agent": _PROBE_SESSION.headers["User-Agent"]}
            if api_key:
                headers["X-API-Key"] = api_key
            probe_url = f"{proxy_prefix}/{target_url}"
            response = requests.get(probe_url, headers=headers, timeout=(5, 12), allow_redirects=True)
        else:
            response = _PROBE_SESSION.get(target_url, timeout=(5, 12), allow_redirects=True)
        if response.status_code >= 400 or response.status_code in {204, 205, 429}:
            return False
        resolved = response.url or target_url
        if _is_entry_or_third_party_url(resolved):
            return False
        content_type = response.headers.get("content-type", "").lower()
        if content_type and not any(kind in content_type for kind in ("text/html", "application/xhtml+xml")):
            return False
        sample = response.text[:120_000].lower()
        # Normal sites (notably GitHub) ship dormant captcha component code in
        # their HTML. Only reject explicit challenge widgets here; Playwright
        # checks the actually visible page text below.
        if any(marker in sample for marker in RAW_HTML_CHALLENGE_MARKERS):
            return False
        if any(marker in sample for marker in ("404 not found", "page not found", "页面不存在", "页面未找到")):
            return False
        # Root-URL pages that resolve to a pure login/registration form are
        # entry pages in disguise (notion.ai, some SaaS dashboards).
        if urlparse(target_url).path.rstrip("/") in ("", "/"):
            title = ""
            lower = sample
            title_start = lower.find("<title")
            if title_start != -1:
                title_open_end = lower.find(">", title_start)
                title_close = lower.find("</title", title_open_end)
                if title_open_end != -1 and title_close != -1:
                    title = lower[title_open_end + 1:title_close]
            login_title_markers = ("登录", "注册", "sign in", "log in", "sign up", "login")
            if title and any(marker in title for marker in login_title_markers):
                return False
        return len(sample.strip()) >= 500
    except requests.RequestException:
        return False


@dataclass
class ScreenshotResult:
    """截图结果"""
    url: str        # 本地文件路径
    alt: str        # 描述文字
    credit: str     # 来源说明
    source_url: str = ""  # Playwright redirects resolved to this page


def _write_provenance(filename: str, target_url: str, source_url: str, description: str) -> None:
    """Persist the page provenance next to the PNG.

    2026-09-24: evidence anchoring requires each article screenshot to be
    traceable back to the page it captured. The PNG alone carries no source
    info, so a sidecar .meta.json keeps target/final URL for downstream
    verification (scripts/rerun_strict.py) and future audits.
    """
    try:
        with open(f"{filename}.meta.json", "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "target_url": target_url,
                    "source_url": source_url or target_url,
                    "description": description,
                    "captured_at": int(time.time()),
                },
                fh,
                ensure_ascii=False,
                indent=2,
            )
    except OSError:
        pass


def _take_screenshot_remote(
    service: str,
    target_url: str,
    description: str,
    width: int,
    height: int,
    clip: dict | None,
) -> ScreenshotResult | None:
    """Call the host-side Playwright service and save the returned PNG.

    The service endpoint is POST {service}/screenshot with JSON
    {"url": ..., "width": ..., "height": ...} -> {"png_base64": ..., "final_url": ...}.
    """
    os.makedirs("data/images", exist_ok=True)
    basename = f"screenshot_{int(time.time())}_{hash(target_url) % 10000:04d}.png"
    filename = f"data/images/{basename}"
    print(f"  [Screenshot] 远程截图: {target_url}")
    try:
        resp = _PROBE_SESSION.post(
            f"{service}/screenshot",
            json={"url": target_url, "width": width, "height": height},
            timeout=(5, 120),
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        print(f"  [Screenshot] 远程截图失败 ({target_url}): {exc}")
        return None
    if not payload.get("png_base64"):
        print(f"  [Screenshot] 远程截图无图像返回 ({target_url}): {str(payload)[:150]}")
        return None
    with open(filename, "wb") as fh:
        fh.write(base64.b64decode(payload["png_base64"]))
    alt = description or urlparse(target_url).netloc
    _write_provenance(filename, target_url, payload.get("final_url") or target_url, description)
    return ScreenshotResult(
        url=filename,
        alt=alt,
        credit=f"Source: {urlparse(payload.get('final_url') or target_url).netloc}",
        source_url=payload.get("final_url") or target_url,
    )


def take_screenshot(
    target_url: str,
    description: str = "",
    width: int = 1280,
    height: int = 800,
    clip: dict | None = None,
) -> ScreenshotResult | None:
    """
    使用 Playwright 截取网页可视区域。

    Args:
        target_url:  要截图的 URL
        description: 图片描述（用于 alt 文字）
        width:       视口宽度，默认 1280
        height:      视口高度，默认 800
        clip:        可选裁剪区域 {"x": 0, "y": 0, "width": 1280, "height": 750}

    Returns:
        ScreenshotResult 或 None（失败时）
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("  [Screenshot] playwright 未安装，请运行: uv pip install playwright && python -m playwright install chromium")
        return None

    # Remote mode: delegate the actual rendering to a Playwright service on
    # the macOS host (SCREENSHOT_HOST_SERVICE=http://host.docker.internal:8919).
    # Chromium inside the Colima VM crashes intermittently on page load
    # (renderer SIGSEGV), while the same binary on the host is stable; the
    # host service also sits outside the container network so the cfp web
    # proxy is reachable exactly as curl sees it.
    host_service = os.environ.get("SCREENSHOT_HOST_SERVICE", "").strip().rstrip("/")
    if host_service:
        return _take_screenshot_remote(host_service, target_url, description, width, height, clip)

    os.makedirs("data/images", exist_ok=True)
    basename = f"screenshot_{int(time.time())}_{hash(target_url) % 10000:04d}.png"
    filename = f"data/images/{basename}"

    print(f"  [Screenshot] 截图: {target_url}")

    if _is_entry_or_third_party_url(target_url):
        print(f"  [Screenshot] 已跳过入口页、登录页或第三方内容页: {target_url}")
        return None

    # Proxy passthrough: when SCREENSHOT_PROXY is set, Chromium loads the
    # reverse-proxied page instead of the raw target. Two proxy shapes are
    # supported:
    #   * ".../api/v1/fetch"  (proxy.qdp.qzz.io gateway): raw target appended,
    #     auth via X-API-Key (SCREENSHOT_PROXY_KEY).
    #   * any prefix ending in "/proxy" (cfp.qdp.qzz.io web proxy): the target
    #     is appended URI-encoded and the handler rewrites every sub-resource
    #     URL in the HTML to stay on the proxy origin — required for full
    #     rendering behind the GFW because the browser never contacts the
    #     target host directly.
    proxy_prefix = os.environ.get("SCREENSHOT_PROXY", "").strip().rstrip("/")
    proxy_key = os.environ.get("SCREENSHOT_PROXY_KEY", "").strip()
    if proxy_prefix:
        if proxy_prefix.endswith("/proxy"):
            fetch_url = f"{proxy_prefix}/{quote(target_url, safe='')}"
        else:
            fetch_url = f"{proxy_prefix}/{target_url}"
        print(f"  [Screenshot] 走代理: {fetch_url[:120]}")
    else:
        fetch_url = target_url

    try:
        with sync_playwright() as p:
            # playwright 1.x default headless launches the chrome-headless-shell
            # binary, which crashes on first request inside Docker (multi-process
            # IPC race in the renderer spawn). Forcing the full chrome binary via
            # executable_path bypasses the crashpad spawn that is missing in
            # headless_shell's bundle layout. --single-process was tried first
            # but breaks as soon as a second browser/page is created in the
            # container; with shm=1gb and the full chrome binary the multi-
            # process layout is stable.
            chrome_exe = os.environ.get(
                "SCREENSHOT_CHROME_EXE",
                "/ms-playwright/chromium-1208/chrome-linux64/chrome",
            )
            launch_kwargs = dict(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-gpu",
                    "--disable-dev-shm-usage",
                ],
            )
            # 上面那个默认路径只存在于容器镜像里。本机（Windows/macOS）照搬
            # 它只会得到 "executable doesn't exist"——路径不存在就让 playwright
            # 自己解析，或者用 SCREENSHOT_CHROME_EXE 显式指定。
            if os.path.exists(chrome_exe):
                launch_kwargs["executable_path"] = chrome_exe
            browser = p.chromium.launch(**launch_kwargs)
            context = browser.new_context()
            if proxy_key:
                # Auth the reverse-proxy request once per browser context. The
                # Worker accepts the X-API-Key header on the top-level GET.
                context.set_extra_http_headers({"X-API-Key": proxy_key})
            page = context.new_page()
            page.set_viewport_size({"width": width, "height": height})

            # DOMContentLoaded is the only wait state used. "load" waits for
            # every sub-resource and behind the web proxy each one costs a
            # separate Worker round-trip; a single slow chunk then trips the
            # whole goto (or worse, crashes the renderer mid-wait). DCL fires
            # as soon as the rewritten HTML is parsed, and the fixed sleep
            # below gives client-side renderers time to paint.
            response = page.goto(fetch_url, wait_until="domcontentloaded", timeout=45000)
            page.wait_for_timeout(4000)

            # A documentation CDN can briefly answer 429 while the browser
            # pool is warming. One delayed retry recovers transient throttling;
            # persistent 429s are rejected as unusable pages.
            if response is not None and response.status == 429:
                page.wait_for_timeout(3000)
                response = page.goto(fetch_url, wait_until="domcontentloaded", timeout=45000)
                page.wait_for_timeout(2500)

            # A successful navigation can still display an HTTP error document.
            if response is not None and response.status >= 400:
                print(f"  [Screenshot] 已跳过 HTTP {response.status} 错误页: {target_url}")
                browser.close()
                return None

            # Redirects can land on an entry page even when the original URL was valid.
            if _is_entry_or_third_party_url(page.url):
                print(f"  [Screenshot] 已跳过重定向后的入口页或第三方内容页: {page.url}")
                browser.close()
                return None

            # Do not publish missing-page, anti-bot, login, CAPTCHA or access-denied pages as article images.
            visible_text = (
                f"{page.title()}\n"
                f"{page.locator('body').inner_text(timeout=5000)}"
            ).lower()
            page_html = page.content().lower()
            if any(marker in visible_text for marker in BLOCKED_PAGE_MARKERS) or any(
                marker in page_html for marker in RAW_HTML_CHALLENGE_MARKERS
            ):
                print(f"  [Screenshot] 已跳过验证或访问拦截页: {target_url}")
                browser.close()
                return None
            if any(marker in visible_text for marker in INCOMPLETE_PAGE_MARKERS):
                print(f"  [Screenshot] 已跳过未完成或暂无内容的页面: {target_url}")
                browser.close()
                return None
            if _is_generic_landing_page(page.url, visible_text):
                print(f"  [Screenshot] 已跳过产品入口或营销落地页: {page.url}")
                browser.close()
                return None
            if page.locator("input[type='password']").count() or (
                "登录" in visible_text and page.locator("input").count() >= 2
            ):
                print(f"  [Screenshot] 已跳过登录或认证表单页: {target_url}")
                browser.close()
                return None

            resolved_url = page.url

            # Capture the substantive feature/document area rather than a
            # generic site header. This also makes pages from the same official
            # documentation host visually distinct for deduplication.
            content_locator = page.locator(
                "main, article, [role='main'], .markdown-body, .prose, .content"
            ).first
            try:
                if content_locator.count() and content_locator.bounding_box():
                    content_locator.scroll_into_view_if_needed(timeout=3000)
                    page.wait_for_timeout(500)
                else:
                    page.mouse.wheel(0, 520)
                    page.wait_for_timeout(500)
            except Exception:
                page.mouse.wheel(0, 520)
                page.wait_for_timeout(500)

            # 截图
            # Chromium can leave page.screenshot waiting forever when a proxy
            # renderer dies after navigation. Bound each capture so one URL
            # cannot stall the whole article batch.
            screenshot_opts: dict = {"path": filename, "timeout": 15000}
            if clip:
                screenshot_opts["clip"] = clip
            else:
                screenshot_opts["full_page"] = False

            page.screenshot(**screenshot_opts)
            if _is_blank_screenshot(filename):
                # The page may have reached networkidle before its client-side
                # renderer painted. Retry the same URL before replacing it.
                print(f"  [Screenshot] 首次截图为空，等待页面继续渲染后重试: {target_url}")
                page.wait_for_timeout(4000)
                page.screenshot(**screenshot_opts)
            browser.close()

        if _is_blank_screenshot(filename):
            os.remove(filename)
            print(f"  [Screenshot] 已跳过空白或未渲染完成的页面: {target_url}")
            return None

        print(f"  [Screenshot] 保存: {filename}")
        _write_provenance(filename, target_url, resolved_url, description)
        alt = description or target_url
        # 返回 API 可访问的 URL，与 upload-image 端点保持一致
        api_url = f"{API_BASE_URL}/api/images/{basename}"
        return ScreenshotResult(
            url=api_url,
            alt=alt,
            credit=description or target_url,
            source_url=resolved_url,
        )

    except Exception as e:
        print(f"  [Screenshot] 截图失败 ({target_url}): {e}")
        return None
