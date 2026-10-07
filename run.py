import time

# ── 修改这两行来切换主题和平台 ──────────────────────────
TOPIC    = "制造企业如何把只存在于老师傅脑中的经验，转化为全组织都能调用的能力"
PLATFORM = "wechat"   # wechat | xiaohongshu | zhihu
# ─────────────────────────────────────────────────────────


def save_article(article: str, platform: str, timestamp: int | None = None) -> str | None:
    """Save a non-empty article and return its path; skip empty results."""
    article = (article or "").strip()
    if not article:
        return None

    stamp = int(time.time()) if timestamp is None else timestamp
    filename = f"output_{platform}_{stamp}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(article + "\n")
    return filename


def main() -> None:
    from agent.graph import run

    print(f"\n🚀 Content Agent 启动")
    print(f"   主题：{TOPIC}")
    print(f"   平台：{PLATFORM}")

    start = time.time()
    result = run(TOPIC, PLATFORM)
    elapsed = time.time() - start
    article = (result.get("article") or "").strip()

    # 打印运行日志
    print(f"\n📋 运行日志（耗时 {elapsed:.1f}s）：")
    for line in result["log"]:
        print(f"   {line}")
    print(f"\n⭐ 最终评分：{result['score']}/10")

    # 打印文章
    print("\n" + "─" * 60)
    print("📝 生成结果：")
    print("─" * 60)
    print(article)
    print("─" * 60)

    # 失败节点会把 final_article 置空。不要把这个状态伪装成成功并写出空文件。
    if not article:
        reason = result["log"][-1] if result.get("log") else "未知原因"
        print(f"\n❌ 未生成文章，未创建空 Markdown 文件。原因：{reason}")
        raise SystemExit(1)

    filename = save_article(article, PLATFORM)
    print(f"\n✅ 已保存到 {filename}")


if __name__ == "__main__":
    main()
