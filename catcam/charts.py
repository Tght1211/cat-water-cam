from __future__ import annotations

import io

# 邮件里要内嵌趋势图：用无界面后端渲染成 PNG 字节，不弹窗、不依赖显示器。
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def trend_png(points: list[tuple[str, int]], title: str) -> bytes:
    """把 [(MM-DD, count), ...] 画成干净的柱状趋势图（邮件内嵌），返回 PNG 字节。

    走 Apple 风的极简：白底、蓝色圆头柱、淡虚线网格、非零柱标值、去掉边框杂线。
    标题/坐标用英文，避开 matplotlib 默认无中文字体导致的方块乱码。
    """
    labels = [p[0] for p in points]
    values = [p[1] for p in points]
    n = len(values)
    fig, ax = plt.subplots(figsize=(6.6, 2.5), dpi=120)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.bar(range(n), values, color="#0a84ff", width=0.62,
                  zorder=3, capstyle="round")
    # 非零柱顶标数值
    vmax = max(values) if values else 0
    for i, v in enumerate(values):
        if v > 0:
            ax.text(i, v + max(1, vmax) * 0.04, str(v), ha="center", va="bottom",
                    fontsize=8, color="#1d1d1f", fontweight="bold")

    ax.set_title(title, fontsize=12, color="#1d1d1f", fontweight="bold", pad=10, loc="left")
    ax.set_ylim(0, max(1, vmax) * 1.28)
    ax.set_yticks([])
    ax.grid(axis="y", color="#e5e5ea", linewidth=1, linestyle=(0, (2, 3)), zorder=0)
    ax.set_xticks(range(n))
    step = max(1, n // 9)
    ax.set_xticklabels(
        [lab if (i % step == 0 or i == n - 1) else "" for i, lab in enumerate(labels)],
        rotation=0, fontsize=8, color="#86868b",
    )
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    return buf.getvalue()
