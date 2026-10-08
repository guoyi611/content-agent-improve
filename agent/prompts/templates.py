from agent.state import Platform
from datetime import date
import re

# ─────────────────────────────────────────────────────────
# 内容方向预设 + 平台 Prompt 模板
#
# 设计原则：
#   "方向"决定写什么（角色、受众、用词风格）
#   "平台"决定怎么写（结构、字数、格式）
#   两者正交组合，互不耦合
#
# 约定：
#   {topic}     → 用户输入的主题
#   {context}   → Researcher 整理好的素材摘要
#   {direction} → 内容方向描述（预设或自定义）
#   [IMAGE: 英文关键词] → ImageFetcher 自动替换（AI 生图/Unsplash）
#   [SCREENSHOT: url, 描述] → ImageFetcher 自动截图（Playwright）
#
# 标题公式：
#   颠覆式 / 方案式 / 悬念式 / 数字式
#   对比式 / 结果前置 / 反问式 / 共情式
# ─────────────────────────────────────────────────────────

# ── 内容方向预设 ──────────────────────────────────────────
# key 是前端传的 direction_id，value 是注入 Prompt 的角色描述
DIRECTION_PRESETS: dict[str, dict] = {
    "tech": {
        "label": "科技 / AI",
        "desc": "科技资讯、AI 动态、产品评测",
        "role": "你是一位资深科技自媒体作者，写作风格：informative、opinionated、not dry。你的文章信息密度高，善用表格对比和代码示例，对技术趋势有自己的判断，不做信息搬运工。关注 AI、开发工具、开源项目、互联网产品。",
    },
    "finance": {
        "label": "财经 / 商业",
        "desc": "商业分析、投资趋势、行业洞察",
        "role": "你是一位财经领域的资深内容创作者，擅长用通俗的语言解读商业逻辑和市场趋势。你的文章特点是：有数据支撑、有独到分析、避免信息茧房。",
    },
    "lifestyle": {
        "label": "生活方式",
        "desc": "好物推荐、效率工具、生活技巧",
        "role": "你是一位生活方式博主，擅长分享提升生活品质的方法和好物。你的风格是：真实体验为主、有审美品味、不浮夸不做作。",
    },
    "education": {
        "label": "知识 / 教育",
        "desc": "学习方法、知识科普、职场成长",
        "role": "你是一位知识分享型创作者，擅长把专业知识讲得通俗有趣。你的文章特点是：逻辑清晰、有实例、让读者有'原来如此'的感觉。",
    },
}

# 默认方向
DEFAULT_DIRECTION = "tech"

# ── 通用写作原则 ──────────────────────────────────────────
WRITING_PRINCIPLES = """
【写作原则】
1. 内容准确：所有数据、事实必须来自素材，禁止编造数据
2. 信息密度高：每一段都要有具体信息（数据、案例、人名、产品名），杜绝空话套话
3. 具体胜过抽象："3天完成"比"很快完成"好，"节省60%时间"比"大幅节省时间"好
4. 保留金句：素材中有力的表述和数据要原样保留，不要弱化
5. 每段只讲一件事，段与段之间有清晰的逻辑衔接
6. 有态度有观点：给出你自己的解读和判断，不做纯粹的信息搬运
7. 中文正文，英文专有名词保留原文（如 Claude、GPT-4o、LangChain）
8. 禁止出现：本文介绍、众所周知、随着XX的发展、让我们一起来看、不用多说、在当今时代

【输出格式——极其重要，必须严格遵守】
- 第一行必须是 # 标题（一级标题，用一个 #）
- 标题下面紧跟一行 > 引用，用一句话概括文章核心观点
- 正文中的小节标题用 ## （二级标题）
- 对比性信息（版本、定价、性能、功能）用表格呈现
- 技术相关内容可以适当展示代码块
- 禁止输出任何结构标签，包括但不限于：
  "标题：""开头钩子：""背景：""核心内容：""观点总结""结尾互动"
  这些是给你的写作指导，不是文章内容，绝对不能出现在输出中
- 直接像一篇发表在平台上的完整文章一样输出，读者看到的就是最终稿
- 只输出文章正文。禁止输出核验计划、能力说明、风险提示、拒绝理由、待办事项或向用户索取材料
- 严禁以"素材不足""信息缺失""无法提取"等理由拒绝写作或输出说明文字。即使素材有限，也必须基于现有素材写出完整文章（可以少写、可以聚焦素材覆盖的部分，但绝不能输出拒绝理由或空内容）
- 不得声称无法联网、无法截图、无法调用工具；配图统一使用规定的占位符交给后续程序处理
- 不得输出任何 API Key、AppSecret、账号凭据或用户提供的操作密钥
- 用 Markdown 格式：# 标题、## 小节、> 引用、**加粗**、列表、表格、代码块等
"""

# ── 证据纪律（2026-09-24 加入：所有主张必须有素材锚点）────
# 根因：旧闸只看截图数量和 critic 分值，放行了"标题断言 X、正文承认没证据"的稿子。
EVIDENCE_DISCIPLINE = """
【证据纪律——每一条都是硬性要求，违反任何一条直接不合格】
1. 标题只能陈述素材能证明的事实。素材里没证实的功能、事件、结论，标题禁止写成已发生的事。
   例：素材只说"README 描述了某功能"→ 标题可以写"README 宣称能…"，禁止写"能一键…了"。
2. 标题里的每一个具体信息（数字、版本号、日期、事件）必须能在素材里找到原文。
   找不到的信息不进标题，也不进正文。
3. 正文每条具体事实（数字、版本号、日期、人物、事件）后面标注来源，格式：
   据〈来源名〉（〈URL〉）。同一来源在本节首次出现时标注，同节重复出现可只写来源名。
4. 每张截图 [SCREENSHOT: URL, 描述] 的 URL 必须来自素材里出现过的真实网址，禁止编造 URL。
   截图所在段落必须直接讨论该 URL 页面的内容——截图与所在段落讨论内容必须一一对应。
   禁例：段落讨论 A 网站的数据，截图却是 B 网站的首页。
5. 素材里找不到、搜索结果里没有的信息，禁止用自己的背景知识补写。没有可核验的最新公开素材时，流程会直接终止，不会给你写作机会；不得生成“常识性定义”或任何兜底文章。
6. 把"已经发生的事"和"公开声明/计划/传闻"严格分开。官方声明用"官方称/官方表示"；
   社区讨论用"有用户反馈/社区讨论认为"；你的推断单独成句并以"我判断/我的判断是"开头。
7. 每节至少有一个"来源锚"：具体来源名 + URL，或一句"公开材料中未查到 XX 信息"的明示。
"""

# ── 中文语言纪律（全平台硬规则）────────
# 小红书过去不注入此段，但审稿端 CRITIC_SYSTEM 的"语言纪律"是无条件应用的，
# 于是 writer 按模板要塞 emoji、critic 按规则判 emoji 违规，三轮重写必挂。
# 来源：human-language-writing skill 的中文禁用清单，转为 LLM 可执行的硬规则。
ZH_DISCIPLINE = """
【中文语言纪律——违反任何一条视为不合格，比字数要求更重要】
1. 标题必须是中文标题。主题里给到的英文产品名/模型名可以保留，但标题必须是中文句子。
   判定口径（与审稿端一致）：中文主语+中文谓语、全句以中文语法骨架为主的标题即算中文标题；
   括注式英文补充（如「ZCode（智谱 AI 编程工具）」）与文件名/路径/URL 片段（如 .git、README、workspace.json）
   不算破坏中文标题；禁止整句英文标题或英文单词超过一半的标题。
2. 正文必须是纯中文句子。允许保留英文的仅限：专有名词（产品、模型、公司、人名）和技术缩写（MCP、RAG、API、GPU、SWE-bench 这类）。整句或整段英文一律禁止；英文引语必须翻成中文转述，用"据 XX 报道"标明出处。
3. 正文禁用一切彩色 emoji（✅⭐🟢⚪🔴💡🔥✨等）。✓ ✗ ⚠ ★ 这类纯文本符号可少量使用。
4. 翻译腔句式禁用："不是…而是…"（正文与标题全文禁用，一次都不许出现；引用素材原话时也必须转述改写，不能照抄）、"这意味着"、"当…的时候"、"诚然…但是…更重要的是"、连续的"被…所…"被动句。
   命中时的改写范式（照此改写，不要自行发挥）：
   反例：企业缺的不是更强的模型，而是能进现场的工程师
   正例：企业真正需要的是能进现场的工程师，更强的模型解决不了落地问题
   反例：它的增长不是因为软件比对手强，而是因为有分发渠道
   正例：它的增长主要来自分发渠道，软件能力只是次要因素
5. 框架词禁用：综上所述、总而言之、值得注意的是、首先…其次…最后、在某种程度上。
6. 大厂黑话禁用：赋能、抓手、闭环、沉淀、对齐、赛道、链路、心智、势能、兜底、底层逻辑、颗粒度。
7. 表格里表示状态用中文词（已支持/不支持/已开源/闭源），禁用 ⭐🟢⚪ 这类符号列状态。
8. 结尾禁止强行升华和展望式套话（未来可期/让我们拭目以待/拥抱XX时代）；用你的独立判断或一个具体问题收尾。
9. 每个数字、版本号、日期、标准号必须来自素材；素材里没有的数据一律不写，宁可少写。"""

# ── 平台 Prompt 模板 ──────────────────────────────────────
PLATFORM_PROMPTS: dict[str, str] = {

    "wechat": """{direction}

请根据下方素材，写一篇关于「{topic}」的公众号推文。

【标题要求】
- 50字以内，核心关键词前置
- 使用中文标点（：、——、？）
- 包含具体信息（版本号、数据、关键特性），让标题本身就有信息量
- 好的标题示例：
  · "Harness Engineering：2026 年真正的门槛在给 AI 搭脚手架"
  · "GLM-5.1：国产大模型编程能力首次逼近 Claude Opus 4.6"
  · "Vibe Coding 正在杀死开源"
- 禁止用：震惊、万字长文、建议收藏、深度好文、你绝对想不到

【结构要求】
1. 标题后紧跟一个 > blockquote，用一句话概括核心观点（这是读者最先看到的信息）
2. 用 ## 分成 2~4 个小节，自然展开：
   - 小节标题要具体有信息量，不要用"背景介绍""详细分析"这类空标题
   - 有对比的内容用表格（版本对比、定价对比、性能跑分等）
   - 技术内容适当展示代码块
   - 每个小节有你的解读，不是纯搬运
3. 结尾给出你的独立判断 + 一个具体问题引导评论（如"你在用什么？欢迎留言"）

【格式要求】
- 字数：1500~2500字
- 语气：informative, opinionated, not dry — 有信息量、有态度、不枯燥
- **加粗核心结论和关键数据**
{image_instructions}
""" + WRITING_PRINCIPLES + EVIDENCE_DISCIPLINE + ZH_DISCIPLINE + """
【素材】
{context}
""",

    "xiaohongshu": """{direction}

请根据下方素材，写一篇关于「{topic}」的小红书笔记。

【标题要求】
- 口语化 + 有具体信息
- 25字以内
- 好的标题举例：
  · "试了3天这个工具，效率直接翻倍"
  · "别再用XX了！这个方案碾压级好用"
- 禁止用：赶紧收藏、建议码住、太全了

【结构要求】
1. 开门见山：第一句话直接说结论或最惊人的信息
2. 3~5个核心要点，每个要点具体（有功能、效果、数据）
3. 一句真实感受 + 一个互动问题

【格式要求】
- 字数：400~600字
- 段落之间空一行
- 结尾另起一行，6~8个 #标签
{image_instructions}
""" + WRITING_PRINCIPLES + ZH_DISCIPLINE + """
【素材】
{context}
""",

    "zhihu": """{direction}

请根据下方素材，写一篇关于「{topic}」的知乎文章。

【标题要求】
- 有信息量、有观点锋芒，关键词前置
- 好的标题示例：
  · "为什么 XX 被严重高估了？"
  · "关于 XX，大多数人理解的都是错的"
  · "如何用 XX 解决 YY 问题？一个被低估的方案"
- 禁止用：浅谈、简述、关于XX的思考、XX的探索与实践

【结构要求】
1. 标题后紧跟 > blockquote 亮出核心观点
2. 用 ## 分成 2~4 个小节，用论据（数据、案例、技术分析）支撑观点
3. 适当加入"反驳常见误解"的角度增加深度
4. 结尾给出独立判断，不是复述前文

【格式要求】
- 字数：1000~2000字
- 语气：理性克制，有独立见解，可以有锋芒但不情绪化
- **加粗核心论点和关键数据**
- 对比性信息用表格，引用关键数据用 > 引用格式
{image_instructions}
""" + WRITING_PRINCIPLES + EVIDENCE_DISCIPLINE + ZH_DISCIPLINE + """
【素材】
{context}
""",
}


# ── 插图模式说明（根据 IMAGE_PROVIDER 注入）──────────────
IMAGE_INSTRUCTION = """- 插图：在每个 ## 段落开始前，单独一行写 [IMAGE: 详细的英文绘图提示词]
  提示词必须包含：① 具体的视觉隐喻（与本段内容强关联）② 色彩方案（含 hex 色值）③ 渲染风格
  可以包含文字标注、品牌视觉元素（logo 轮廓、品牌色）、数据可视化。
  ✗ [IMAGE: AI technology]（太抽象）
  ✗ [IMAGE: performance comparison]（没有画面）
  ✓ [IMAGE: A futuristic digital illustration showing a glowing Chinese dragon made of circuit board patterns and code streams, coiling around a large glowing "5.1" number. Blue and gold light emanating outward. Color palette: deep navy (#0A1628), electric blue (#1A73E8), gold (#FFD600). Style: flat vector with cinematic lighting, 16:9]"""

SCREENSHOT_INSTRUCTION = """- 插图：在需要展示产品界面、官方文档或实际效果的论点前，单独一行写 [SCREENSHOT: 官方网址, 中文描述]
  截图要服务于它上方那一段的论证，中文描述写清这张图具体展示什么。
  URL 只能取自「素材摘要」中列出的页面，禁止凭记忆编造——编造的 URL 会 404 导致截图失败。
  素材里没有与段落对应的页面时，这一段就不插图。图片数量由内容需要决定，不要为了凑数硬配。
  不要官网首页、搜索结果页、聚合站；优先官方文档页、功能子页、结果页。
  同一个页面只用一次。
  ✓ [SCREENSHOT: https://platform.openai.com/docs/guides/text, 官方文本功能文档]"""

MIXED_INSTRUCTION = """- 插图方式一（截图）：在需要展示**产品界面、官方页面、实际效果**的段落前，单独一行写 [SCREENSHOT: 官方网址, 中文描述]
  截图目标必须是与本段内容直接相关的官方网站、产品页面、文档页面。
  ✓ [SCREENSHOT: https://www.anthropic.com/claude, Claude 产品介绍页]
- 插图方式二（AI 生图）：在需要**概念性、装饰性、数据可视化**插图的段落前，单独一行写 [IMAGE: 详细的英文绘图提示词]
  提示词必须包含具体的视觉隐喻、色彩方案（含 hex 色值）、渲染风格。
  ✓ [IMAGE: A futuristic digital illustration showing neural network nodes connected by glowing pathways. Color palette: deep navy (#0A1628), electric blue (#1A73E8), gold (#FFD600). Style: flat vector with cinematic lighting, 16:9]
- 根据内容特点灵活选择：有官方页面可截的用 SCREENSHOT，需要创意表达的用 IMAGE"""


def get_direction_text(direction: str) -> str:
    """
    获取方向描述文本。
    如果是预设 key（如 "tech"），返回预设的 role；
    否则视为自定义方向描述，直接使用。
    """
    if direction in DIRECTION_PRESETS:
        return DIRECTION_PRESETS[direction]["role"]
    # 自定义方向：用户直接输入的描述
    return f"你是一位专业的内容创作者。你的写作方向是：{direction}"


def _get_image_instruction(image_mode: str) -> str:
    """根据配图模式返回对应的插图说明"""
    if image_mode == "screenshot":
        return SCREENSHOT_INSTRUCTION
    elif image_mode == "mixed":
        return MIXED_INSTRUCTION
    else:
        return IMAGE_INSTRUCTION


def build_prompt(platform: Platform, topic: str, context: str, direction: str = "", outline: str = "", image_mode: str = "image", critic_feedback: str = "") -> str:
    """用实际内容替换模板中的占位符"""
    if not direction:
        direction = DEFAULT_DIRECTION
    direction_text = get_direction_text(direction)
    template = PLATFORM_PROMPTS[platform]

    # 替换插图说明
    image_instruction = _get_image_instruction(image_mode)
    template = template.replace("{image_instructions}", image_instruction)

    # Post-processing, publishing and credential instructions are handled by
    # application code and must not leak into the article-writing prompt.
    article_topic = re.split(r"完成以上任务后执行以下任务", topic, maxsplit=1)[0].strip()
    article_topic = re.sub(r"sk-[A-Za-z0-9_-]{16,}", "[已隐藏密钥]", article_topic)
    article_topic = re.sub(r"(?i)(appsecret\s*(?:是|=|:)\s*)\S+", r"\1[已隐藏]", article_topic)

    result = (
        template
        .replace("{direction}", direction_text)
        .replace("{topic}", article_topic)
        .replace("{context}", context)
    )
    result = f"【当前日期（仅供时间判断）】{date.today().isoformat()}。除非素材明确提供该日期，否则禁止把此日期或任何由它推导出的日期写入文章正文。\n\n{result}"
    # 注入文章规划（如果有）
    if outline:
        result = result.replace("【素材】", f"【文章规划（Planner 产出，请参考但不必死板遵循）】\n{outline}\n\n【素材】")
    # 重写轮次：把上一轮评审的具体问题顶到【素材】正前方。
    # 只补搜素材是不够的——writer 拿不到 feedback 就会重复上一轮的违规，
    # 三轮都在同样的硬闸上翻车。
    if critic_feedback:
        result = result.replace(
            "【素材】",
            f"【上一轮评审未通过，本次必须逐条修正以下问题，不得重复】\n{critic_feedback}\n\n【素材】",
        )
    return result
