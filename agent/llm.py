"""
LLM 工厂模块
统一读取 .env 配置，返回对应的 LLM 实例。

支持两种规范：
  - openai    兼容 OpenAI Chat Completions API（Kimi、DeepSeek、通义等均支持）
  - anthropic 原生 Anthropic API（Claude 系列）

.env 配置示例：

  # 使用 OpenAI 官方
  LLM_PROVIDER=openai
  LLM_API_KEY=sk-xxxx
  LLM_BASE_URL=https://api.openai.com/v1   # 可省略，这是默认值
  LLM_MODEL=gpt-4o-mini

  # 使用 Anthropic 官方
  LLM_PROVIDER=anthropic
  LLM_API_KEY=sk-ant-xxxx
  LLM_MODEL=claude-3-5-haiku-20241022
"""

import os
from functools import lru_cache
from langchain_core.language_models import BaseChatModel
from agent.config import get_config


@lru_cache(maxsize=8)
def get_llm(role: str = "default") -> BaseChatModel:
    """
    根据配置返回 LLM 实例（优先级：env > SQLite > 默认值）。
    结果按 role 缓存。

    role="default" 读 LLM_* 配置；其他角色（如 "writer"）先读
    <ROLE>_LLM_*，缺哪一项就回退到 LLM_* 的同名配置——这样可以只给某个
    节点换模型，其余节点不受影响。
    """
    prefix = "LLM" if role == "default" else f"{role.upper()}_LLM"
    label = "LLM" if role == "default" else f"LLM:{role}"

    provider = (
        get_config(f"{prefix}_PROVIDER") or get_config("LLM_PROVIDER", "openai")
    ).lower()
    api_key = get_config(f"{prefix}_API_KEY") or get_config("LLM_API_KEY")
    model = get_config(f"{prefix}_MODEL") or get_config("LLM_MODEL", "gpt-4o-mini")
    base_url = get_config(f"{prefix}_BASE_URL") or get_config("LLM_BASE_URL")
    reasoning_effort = (
        get_config(f"{prefix}_REASONING_EFFORT") or get_config("LLM_REASONING_EFFORT")
    )

    if not api_key:
        raise ValueError(f"未设置 {prefix}_API_KEY 或 LLM_API_KEY，请在设置中填写")

    if provider == "anthropic":
        return _make_anthropic(api_key, model, label)
    elif provider == "openai":
        return _make_openai(api_key, model, base_url, reasoning_effort, label)
    else:
        raise ValueError(
            f"不支持的 {prefix}_PROVIDER: '{provider}'，"
            "请设置为 'openai' 或 'anthropic'"
        )


def _make_openai(
    api_key: str,
    model: str,
    base_url: str | None = None,
    reasoning_effort: str | None = None,
    label: str = "LLM",
) -> BaseChatModel:
    """
    创建兼容 OpenAI 规范的 LLM。
    Kimi、DeepSeek、通义、硅基流动等只需改 base_url 和 model 即可。

    base_url 由 get_llm 按 role 解析后传入，这里不再回读通用配置——
    否则 writer 配了独立端点时，下面的 extra_body 分支会看错来源。
    """
    from langchain_openai import ChatOpenAI

    # Relay providers can take several minutes to return a long reasoning
    # response. The previous 120-second request timeout converted a slow but
    # healthy response into LangChain's generic APIConnectionError. Keep the
    # connect timeout short while allowing a bounded long read.
    try:
        read_timeout = float(get_config("LLM_READ_TIMEOUT_SECONDS", "600"))
    except ValueError:
        read_timeout = 600.0
    try:
        connect_timeout = float(get_config("LLM_CONNECT_TIMEOUT_SECONDS", "30"))
    except ValueError:
        connect_timeout = 30.0
    # openai SDK 会对 httpx.ReadTimeout 自动整请求重发（_base_client.py 里
    # TimeoutException → continue），max_retries=3 意味着一次死流最坏
    # 4×read_timeout 才抛出。读超时说明流已死，SDK 级重试只会把挂死时间
    # 乘 4；重试交由上层（writer 的 stream→invoke 兜底、graph 的 critic
    # 重写循环）自己做，这里固定为 0。
    max_retries = 0

    kwargs = dict(
        api_key=api_key,
        model=model,
        max_retries=max_retries,
        # langchain-openai 1.x 的字段名是 request_timeout（pydantic 字段）。
        # 之前传 timeout= 会被 pydantic 静默丢弃，超时从未生效，
        # 流式响应在中间链路断流时会无限期挂死。
        request_timeout=(connect_timeout, read_timeout),
    )
    if base_url:
        kwargs["base_url"] = base_url
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    # 本地 litellm 网关的 router 默认 per-attempt timeout 是 15s（config.yaml
    # router_settings.timeout，为 auto 组快失败而设）。直连模型组（如
    # GPT-5.6-Terra）的长文生成首字节常超 15s，网关会以 litellm.Timeout
    # 408 拒绝。请求体里的 timeout 字段可覆盖该默认（litellm proxy 读取
    # body 级 timeout），因此把读超时同样传给网关，让单次 attempt 的
    # 预算与客户端读超时一致。
    if base_url:
        kwargs["extra_body"] = {"timeout": read_timeout}
        # Qwen3 系部署默认开思考模式：首字节前先烧 30-90s reasoning token，
        # 6000+ 字长文生成总时长轻松破 600s 读超时，流式心跳等不到正文
        # chunk 就判死。chat_template_kwargs 是 vLLM/SGLang 标准透传，
        # enable_thinking=false 直接关思考，实测同 prompt 86s→33s。对不支持
        # 该参数的上游为无害多余字段，网关/模型自动忽略。
        kwargs["extra_body"]["chat_template_kwargs"] = {"enable_thinking": False}
        # 部分上游部署前置 Cloudflare（524 = origin 120s 无完整响应）。非流式
        # 长生成必然触发；流式让字节持续流动即可绕开。streaming=True 使
        # .invoke() 也走 SSE 聚合，与 .stream() 同路径。
        kwargs["streaming"] = True

    print(f"  [{label}] OpenAI 规范 | model={model} | base_url={base_url or '(default)'}")
    return ChatOpenAI(**kwargs)


def _make_anthropic(api_key: str, model: str, label: str = "LLM") -> BaseChatModel:
    """创建原生 Anthropic LLM。"""
    from langchain_anthropic import ChatAnthropic

    print(f"  [{label}] Anthropic 规范 | model={model}")
    return ChatAnthropic(
        api_key=api_key,
        model_name=model,
    )


def reset_llm_cache() -> None:
    """设置更新后调用此函数，使 LLM 实例缓存失效，下次调用时重新创建。"""
    get_llm.cache_clear()
