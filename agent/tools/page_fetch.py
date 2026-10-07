"""Shared helpers for fetching and extracting readable web page text."""

from urllib.parse import quote

import requests
import trafilatura

from agent.config import get_config


DEFAULT_RESEARCH_PROXY = "https://cfp.qdp.qzz.io/proxy"
DEFAULT_CONNECT_TIMEOUT = 5.0
DEFAULT_READ_TIMEOUT = 15.0
USER_AGENT = "Mozilla/5.0 (content-agent research)"


def _timeout_setting(name: str, default: float) -> float:
    try:
        value = float(get_config(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(1.0, value)


def _targets(url: str) -> list[tuple[str, str]]:
    proxy = get_config("RESEARCH_PROXY_URL", DEFAULT_RESEARCH_PROXY).strip().rstrip("/")
    targets: list[tuple[str, str]] = []
    if proxy:
        targets.append((f"{proxy}/{quote(url, safe='')}", "proxy"))
    targets.append((url, "direct"))
    return targets


def fetch_readable_text(url: str) -> tuple[str, str]:
    """Return ``(text, error)`` while keeping page failures non-fatal.

    Search snippets are still useful evidence, so a blocked page must not
    abort the whole generation run. The error string is intended for logs.
    """
    if not url or url.startswith(("javascript:", "data:")):
        return "", "跳过非网页 URL"

    timeout = (
        _timeout_setting("RESEARCH_CONNECT_TIMEOUT_SECONDS", DEFAULT_CONNECT_TIMEOUT),
        _timeout_setting("RESEARCH_READ_TIMEOUT_SECONDS", DEFAULT_READ_TIMEOUT),
    )
    errors: list[str] = []

    for target, label in _targets(url):
        try:
            response = requests.get(
                target,
                headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
                timeout=timeout,
                allow_redirects=True,
            )
            response.raise_for_status()
            text = trafilatura.extract(
                response.text,
                include_links=True,
                include_tables=True,
            )
            text = (text or "").strip()
            if len(text) >= 200:
                return text, ""
            errors.append(f"{label}: 正文提取为空")
        except requests.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            suffix = f", HTTP {status}" if status else ""
            errors.append(f"{label}: {type(exc).__name__}{suffix}")
        except Exception as exc:  # noqa: BLE001 - extraction must not stop research
            errors.append(f"{label}: {type(exc).__name__}")

    return "", "; ".join(errors) or "未找到可用正文"
