from langchain_core.messages import HumanMessage
from agent.state import AgentState
from agent.prompts.templates import build_prompt
from agent.llm import get_llm
from agent.config import get_config
from agent.tools.style_guard import strip_translationese
import re

_SCREENSHOT_PLACEHOLDER = re.compile(r"^\[SCREENSHOT:\s*(\S+?)\s*,", re.M)

# narrative_design 改动 2：解析占位符 + 元数据
# 按 kind 分流：IMAGE 整体吞下 prompt 内容（hex 色值常带逗号，不能按 ", " 拆）；
# SCREENSHOT 按 "URL, 中文描述" 切分（固定两段）。
_IMAGE_RE = re.compile(r"^\[IMAGE:\s*(.+?)\]\s*$", re.M)
_SCREENSHOT_FULL_RE = re.compile(
    r"^\[SCREENSHOT:\s*(\S+?)\s*,\s*(.+?)\]\s*$", re.M
)
_SCREENSHOT_FALLBACK_RE = re.compile(r"^\[SCREENSHOT:\s*(\S+?)\s*\]?\s*$", re.M)
_PLACEHOLDER_ANY_RE = re.compile(
    r"^\[(IMAGE|SCREENSHOT):", re.M
)
_HEADING_RE = re.compile(r"(?m)^##\s+.+$")
_ROLE_NOTE_RE = re.compile(
    r"\(\s*role\s*:\s*(\w+)\s*(?:;\s*reason\s*:\s*(.+?))?\s*\)"
)
_VALID_ROLES = {"hero", "evidence", "contrast", "transition", "decorative"}


def _extract_image_placeholders(draft: str) -> list[dict]:
    """解析 draft 中所有 IMAGE / SCREENSHOT 占位符的逐项元数据（narrative_design 改动 2）。

    字段见 AgentState.image_placeholders 注释。
    解析失败（正则不匹配）→ 字段为空列表，下游按"无元数据"降级。

    按 kind 分流匹配：
      IMAGE       → [IMAGE: <prompt 整段>]         prompt 内可能含 ","（hex 色值）
      SCREENSHOT  → [SCREENSHOT: <url>, <描述>]    固定两段，按 ", " 拆
    """
    if not draft:
        return []

    headings = [(m.start(), m.group(0).strip()) for m in _HEADING_RE.finditer(draft)]
    out: list[dict] = []

    for line_m in _PLACEHOLDER_ANY_RE.finditer(draft):
        line_start = line_m.start()
        # 占位符占整行；从 line_start 读至行尾（或 draft 末尾）
        line_end = draft.find("\n", line_start)
        if line_end < 0:
            line_end = len(draft)
        line_text = draft[line_start:line_end]

        kind_token = line_m.group(1).lower()
        kind = "image" if kind_token == "image" else "screenshot"
        caption: str | None = None

        if kind == "image":
            # IMAGE 整段吞下（避免 prompt 内的 "," 被误切为 caption）
            m = _IMAGE_RE.match(line_text)
            if not m:
                # 格式异常（如括号未闭合）→ 跳过该占位符，避免污染下游
                continue
        else:
            m = _SCREENSHOT_FULL_RE.match(line_text)
            if m:
                caption = m.group(2).strip()
            else:
                # 退化：URL 后无描述
                fb = _SCREENSHOT_FALLBACK_RE.match(line_text)
                if not fb:
                    continue

        # 找 line_start 之前最近的 heading
        heading: str | None = None
        heading_offset = -1
        for h_offset, h_text in headings:
            if h_offset <= line_start:
                heading_offset = h_offset
                heading = h_text
            else:
                break

        # paragraph_index：在该 heading 段内占位符前有几个空行分段（粗略）
        para_idx = 0
        if heading_offset >= 0:
            section = draft[heading_offset:line_start]
            para_idx = section.count("\n\n")

        # 占位符下一行的 (role: <r>; reason: <text>) 自注（LLM 可选输出）
        role, reason = "unassigned", ""
        nxt_start = line_end + 1
        nxt_end = draft.find("\n", nxt_start)
        if nxt_end < 0:
            nxt_end = len(draft)
        if nxt_start < len(draft):
            nxt_line = draft[nxt_start:nxt_end].strip()
            note_m = _ROLE_NOTE_RE.search(nxt_line)
            if note_m:
                r = note_m.group(1).lower()
                role = r if r in _VALID_ROLES else "unassigned"
                reason = (note_m.group(2) or "").strip()

        out.append(
            {
                "placeholder": line_text,
                "kind": kind,
                "heading": heading,
                "paragraph_index": para_idx,
                "char_offset": line_start,
                "role": role,
                "reason": reason,
                "caption": caption,
            }
        )

    return out


def _drop_unanchored_screenshots(draft: str) -> tuple[str, int]:
    """删除不被正文引用自证的截图占位符。

    保留条件（对应 SCREENSHOT_INSTRUCTION 的锚定规则）：
    占位符里的 URL 必须在占位符之外的正文里出现过——图是被引用的页面本身。
    位置不设限：紧跟引用句、放在段落末尾或小节标题之后都可以，
    评判标准只有"这个页面是否被正文当作证据引用了"。
    """
    lines = draft.split("\n")
    body_without_placeholders = "\n".join(
        line for line in lines if not line.strip().startswith("[SCREENSHOT:")
    )

    keep: list[str] = []
    dropped = 0
    for line in lines:
        m = _SCREENSHOT_PLACEHOLDER.match(line.strip())
        if not m:
            keep.append(line)
            continue
        url = m.group(1).rstrip("/")
        if url not in body_without_placeholders:
            dropped += 1
            continue
        keep.append(line)
    return "\n".join(keep), dropped


def _get_image_mode_from_provider() -> str:
    """根据 IMAGE_PROVIDER 配置决定插图模式（仅作为 article_type 缺省时的降级路径）"""
    provider = get_config("IMAGE_PROVIDER").lower()
    if provider == "screenshot":
        return "screenshot"
    if provider == "mixed":
        return "mixed"
    return "image"


def _resolve_image_mode(state: AgentState) -> str:
    """按 narrative_design 改动 1 的路由决定插图模式。planner 输出的 article_type
    决定主路径；缺省时降级到 IMAGE_PROVIDER 旧 image_mode（老文章兼容）。"""
    article_type = state.get("article_type")
    if article_type in ("consulting", "creative"):
        has_citations = bool(state.get("screenshot_source_urls"))
        if article_type == "consulting":
            return "screenshot"
        return "mixed" if has_citations else "image"
    # 兜底：article_type 缺失（planner 旧版本 / 解析失败）
    return _get_image_mode_from_provider()


def writer_node(state: AgentState) -> dict:
    """
    根据平台 Prompt 模板和素材生成初稿。
    初稿中包含 [IMAGE: 英文关键词] 或 [SCREENSHOT: url, 描述] 占位符，
    由后续的 image_fetcher_node 替换成真实图片。
    """
    print(f"\n[Writer] 开始写作（平台：{state['platform']}）...")

    article_type = state.get("article_type")
    has_citations = bool(state.get("screenshot_source_urls"))
    image_mode = _resolve_image_mode(state)
    prompt = build_prompt(
        platform=state["platform"],
        topic=state["topic"],
        context=state["context"],
        direction=state.get("direction", ""),
        outline=state.get("outline", ""),
        image_mode=image_mode,
        critic_feedback=state.get("critic_feedback", ""),
        article_type=article_type,
        has_citations=has_citations,
    )
    # 应用层超时（对抗"活流挂死"）：httpx 的 read timeout 只在 socket 层无字节时
    # 触发，而网关/上游在卡死时仍持续发 SSE 心跳字节，socket 永远有数据 →
    # read timeout 形同虚设（实测 5s 超时配置 3.5 分钟不触发）。因此在这里做两层
    # 应用层看门狗：整体 deadline（W_TOTAL，默认 25 分钟）+ 无内容心跳窗口
    # （W_IDLE，默认 300 秒——收到 chunk 但迟迟没有携带正文内容的 chunk）。
    import time as _time
    w_total = float(get_config("WRITER_TOTAL_TIMEOUT_SECONDS", "1500"))
    w_idle = float(get_config("WRITER_IDLE_TIMEOUT_SECONDS", "300"))

    class _StreamWatchdog(Exception):
        pass

    def _stream_with_deadline():
        chunks: list[str] = []
        started = _time.monotonic()
        last_content = started
        for chunk in get_llm("writer").stream([HumanMessage(content=prompt)]):
            now = _time.monotonic()
            if now - started > w_total:
                raise _StreamWatchdog(
                    f"writer 流式整体超时 {now - started:.0f}s > {w_total:.0f}s"
                )
            if now - last_content > w_idle:
                raise _StreamWatchdog(
                    f"writer 流式心跳超时：{now - last_content:.0f}s 无正文 chunk"
                )
            if chunk.content:
                chunks.append(chunk.content)
                last_content = now
        return "".join(chunks)

    try:
        # 流式消费：容器经 socat/vmnet 转发到宿主机网关，非流式长生成会在
        # 中间链路静默断链（连接看似 ESTABLISHED 但字节永不到达）。流式让
        # 字节持续流动，避免僵尸连接；聚合结果与 invoke 等价。
        text = _stream_with_deadline()
        draft = text.strip() if text else get_llm("writer").invoke([HumanMessage(content=prompt)]).content.strip()
    except TypeError as e:
        if "null value for 'choices'" in str(e):
            print("  ⚠️ LLM 返回空响应（可能触发内容审核），正在重试...")
            res = get_llm("writer").invoke([HumanMessage(content=prompt)])
            draft = res.content.strip()
        else:
            raise
    except _StreamWatchdog as wd:
        # 看门狗触发说明流已挂死（有 SSE 心跳但无正文）。invoke 走非流式路径，
        # 由网关直接返回完整结果，绕开卡死的流式链路。
        print(f"  ⚠️ {wd}，回退到 invoke 重试一次...")
        res = get_llm("writer").invoke([HumanMessage(content=prompt)])
        draft = res.content.strip()
    except TypeError as e:
        if "null value for 'choices'" in str(e):
            print("  ⚠️ LLM 返回空响应（可能触发内容审核），正在重试...")
            res = get_llm("writer").invoke([HumanMessage(content=prompt)])
            draft = res.content.strip()
        else:
            raise
    except Exception as stream_exc:
        # LiteLLM 网关偶尔在长响应的流式传输中途断开（TransferEncodingError
        # "Not enough data to satisfy transfer length header"），导致 stream
        # 迭代器抛 APIError。上游字节已部分到达、聚合结果不可信，直接重试
        # 一次 invoke 是当前最稳的恢复路径。
        # 注意：openai SDK 的超时异常是 APITimeoutError（继承 APIConnectionError，
        # 不含连续子串 "APIError"），按类继承判断而不是子串匹配。
        from openai import APIError, APIConnectionError
        if isinstance(stream_exc, (APIError, APIConnectionError)):
            err_name = type(stream_exc).__name__
            print(f"  ⚠️ 流式中断（{err_name}），回退到 invoke 重试一次...")
            res = get_llm("writer").invoke([HumanMessage(content=prompt)])
            draft = res.content.strip()
        else:
            raise

    # 截图数量不再强制。此前是"不足 12 张就完整重写全文"，结果是 writer 为了
    # 凑数牺牲正文（文章明显变短），并硬配与论点无关的页面（图不对应）。现在改
    # 由 SCREENSHOT_INSTRUCTION 要求"有对应页面才插图"，数量随内容需要决定；
    # 截图失败的部分由 image_fetcher 自动从正文移除，本就不需要预留候选。

    # narrative_design 改动 2：解析占位符元数据（heading / paragraph_index / role / reason）
    # —— 设计要求在 _drop_unanchored_screenshots 之前完成，让元数据包含 writer 实际
    # 输出的全部占位符（含未锚定的）；下游 refiller / critic 按需过滤。
    image_placeholders = _extract_image_placeholders(draft)

    # 截图锚定的确定性执法：prompt 只能约束大概率，writer 手头真引用不够时
    # 仍会拿无关页面凑数。一张截图要保留，唯一条件是它的 URL 出现在
    # 占位符之外的正文里（即被当作证据引用过）；位置不限。
    draft, dropped_shots = _drop_unanchored_screenshots(draft)
    if dropped_shots:
        print(f"  [Writer] 移除未锚定截图 {dropped_shots} 张（URL 不在正文引用中）")

    # 消除翻译腔句式：prompt 层面的禁令压不住模型的语言惯性，
    # 这里在交付 critic 之前做一次确定性检测 + 局部重写，只动命中的行。
    draft, changed_lines = strip_translationese(draft)
    if changed_lines:
        print(f"  [StyleGuard] 通读改写 {changed_lines} 行")

    # 统计占位符数量，方便调试
    import re
    image_count = len(re.findall(r"\[IMAGE:", draft))
    screenshot_count = len(re.findall(r"\[SCREENSHOT:", draft))
    total = image_count + screenshot_count

    mode_label = {"screenshot": "截图", "mixed": "混合", "image": "AI 生图"}.get(image_mode, image_mode)
    detail = f"AI 生图 {image_count}" if image_count else ""
    if screenshot_count:
        detail = f"{detail + '，' if detail else ''}截图 {screenshot_count}"
    print(f"  初稿完成（{len(draft)}字，插图模式：{mode_label}，占位符 {total} 个：{detail}）")

    return {
        "draft": draft,
        "image_placeholders": image_placeholders,
        "log": state.get("log", []) + [
            f"✍️ {state['platform']} 初稿完成（{len(draft)}字）"
        ],
    }
