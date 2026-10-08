from langchain_core.messages import HumanMessage
from agent.state import AgentState
from agent.prompts.templates import build_prompt
from agent.llm import get_llm
from agent.config import get_config
from agent.tools.style_guard import strip_translationese
import re


def _get_image_mode() -> str:
    """根据 IMAGE_PROVIDER 配置决定插图模式"""
    provider = get_config("IMAGE_PROVIDER").lower()
    if provider == "screenshot":
        return "screenshot"
    if provider == "mixed":
        return "mixed"
    return "image"


def writer_node(state: AgentState) -> dict:
    """
    根据平台 Prompt 模板和素材生成初稿。
    初稿中包含 [IMAGE: 英文关键词] 或 [SCREENSHOT: url, 描述] 占位符，
    由后续的 image_fetcher_node 替换成真实图片。
    """
    print(f"\n[Writer] 开始写作（平台：{state['platform']}）...")

    image_mode = _get_image_mode()
    prompt = build_prompt(
        platform=state["platform"],
        topic=state["topic"],
        context=state["context"],
        direction=state.get("direction", ""),
        outline=state.get("outline", ""),
        image_mode=image_mode,
        critic_feedback=state.get("critic_feedback", ""),
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
        "log": state.get("log", []) + [
            f"✍️ {state['platform']} 初稿完成（{len(draft)}字）"
        ],
    }
