from typing import TypedDict, Literal, NotRequired

# 支持的平台类型
Platform = Literal["wechat", "xiaohongshu", "zhihu"]


class AgentState(TypedDict):
    """
    贯穿所有节点的共享状态。
    每个节点只负责填充自己那部分，数据自动向下游传递。

    数据流向：
      pre_researcher → 填 raw_materials, history_context（预搜索）
      planner        → 填 outline（基于搜索结果规划）
      researcher     → 填 keywords, raw_materials, context（精准补充搜索）
      writer         → 填 draft（含 [IMAGE:...] 占位符）
      paraphraser    → 改写 draft 并恢复 Markdown/HTML 结构
      image_fetcher  → 填 images, final_article
    """
    topic: str                   # 用户输入的主题
    platform: Platform           # 目标平台
    direction: str               # 内容方向（预设 key 或自定义描述）
    outline: str                 # Planner 规划的文章大纲（角度、框架、要点、视觉叙事线）
    article_type: NotRequired[Literal["consulting", "creative"]]  # Planner 分类：咨询类（评测/对比/指南，默认）/ 创造类（观点/叙事/概念）。决定 writer prompt 分支与 image_fetcher provider 路由
    keywords: list[str]          # Planner 拆解的搜索关键词
    raw_materials: list[str]     # 搜索到的原始素材（每条一个字符串）
    context: str                 # 整理后的素材摘要（喂给 Writer）
    draft: str                   # 初稿，含 [IMAGE: 英文关键词] 占位符
    paraphrase_applied: NotRequired[bool]  # 是否已通过指定改写接口
    paraphrase_request_count: NotRequired[int]  # 本次实际调用改写接口次数
    paraphrase_changed_count: NotRequired[int]  # 接口返回发生变化的文本片段数
    paraphrase_request_ids: NotRequired[list[str]]  # 服务端返回的请求凭证
    paraphrase_error: NotRequired[str]  # 改写失败原因（不包含密钥）
    images: dict[str, str]       # { "[IMAGE: xxx]": "![alt](url)\n*credit*" }
    final_article: str           # 替换图片占位符后的最终文章
    log: list[str]               # 运行日志，供前端侧边栏展示
    history_context: str         # 从向量库检索到的历史素材
    critic_score: float          # Critic 评分（1~10；narrative_design 改动 3 起支持 0.5 颗粒：LLM 4 项 9 分制 + 代码算图文一致性 1 分 = 10 分制；pass ≥ 7.5）
    critic_feedback: str         # Critic 修改建议
    retry_action: NotRequired[str]  # 重试路由：rewrite=直接重写 / research=补搜后重写
    retry_count: int             # 重写次数（最多 2 次）
    image_style: NotRequired[str]  # 图片风格预设（可选）
    topic_id: NotRequired[int]     # 数据库 topic ID，用于关联向量库
    screenshot_retry_count: NotRequired[int]  # 截图实际成功数量不足时的补偿轮次（最多 4 轮）
    screenshot_retry_note: NotRequired[str]   # 补偿写作时避开的失败 URL
    screenshot_success_count: NotRequired[int]  # Playwright 实际成功截图数
    needs_screenshot_retry: NotRequired[bool]  # 是否需要重新生成截图候选
    screenshot_source_urls: NotRequired[list[str]]  # 已接受截图的最终页面 URL
    screenshot_attempted_urls: NotRequired[list[str]]  # 本次生成已尝试过的候选，跨补图轮次去重
    research_failed: NotRequired[bool]  # 没有可核验的本轮搜索素材时硬性终止
    image_placeholders: NotRequired[list[dict]]
    # Writer 节点解析的 IMAGE/SCREENSHOT 占位符逐项元数据（narrative_design 改动 2）。
    # 每项：
    #   placeholder: 原始占位符字符串（如 "[IMAGE: ...]"）
    #   kind: "image" | "screenshot"
    #   heading: 占位符前最近的 `##` 标题（首屏前则为 None）
    #   paragraph_index: 在该 heading 下是第几个段落（从 0 计）
    #   char_offset: 占位符在 draft 里的起始字符偏移
    #   role: LLM 自注角色（hero/evidence/contrast/transition/decorative/unassigned）
    #   reason: LLM 自注「为什么放这里」
    #   caption: SCREENSHOT 第二组（中文描述），IMAGE 则为 None
    # 消费方：refiller 按元数据找未覆盖 heading 补图；critic 检查自注完整性；
    #        image_fetcher 按 article_type 路由。解析失败 → 字段为空列表 → 降级到旧行为。
