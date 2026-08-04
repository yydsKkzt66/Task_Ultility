from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "doc_comparison"
OUTPUT_PATH = OUTPUT_DIR / "planner_comparison_doc_dashboard.png"


def pick_font() -> FontProperties:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "FangSong",
        "STFangsong",
        "DejaVu Sans",
    ]
    for name in candidates:
        try:
            return FontProperties(family=name)
        except Exception:
            continue
    return FontProperties()


FONT = pick_font()
plt.rcParams["axes.unicode_minus"] = False


def add_panel(ax, title: str, facecolor: str = "#FFFDF8") -> None:
    ax.set_axis_off()
    patch = FancyBboxPatch(
        (0.01, 0.01),
        0.98,
        0.98,
        boxstyle="round,pad=0.018,rounding_size=0.03",
        linewidth=1.4,
        edgecolor="#CCBFA3",
        facecolor=facecolor,
        transform=ax.transAxes,
        zorder=0,
    )
    ax.add_patch(patch)
    ax.text(
        0.04,
        0.93,
        title,
        transform=ax.transAxes,
        fontproperties=FONT,
        fontsize=16,
        fontweight="bold",
        color="#2E3A2F",
        va="top",
    )


def add_bullets(
    ax,
    items: list[str],
    start_y: float = 0.82,
    size: int = 11,
    step: float = 0.11,
    width: int = 36,
) -> None:
    y = start_y
    for item in items:
        ax.text(
            0.05,
            y,
            fill(f"• {item}", width=width, break_long_words=False),
            transform=ax.transAxes,
            fontproperties=FONT,
            fontsize=size,
            color="#29323A",
            va="top",
            linespacing=1.35,
        )
        y -= step


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(20, 13.5), facecolor="#F4F1EA")
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.02, 1.25, 1.08],
        width_ratios=[1.05, 1.05, 0.9],
        hspace=0.16,
        wspace=0.12,
    )

    title = "《粗略对比.txt》核心内容可视化总览"
    subtitle = (
        "目标不是证明 Contextual Bandit 在所有任务上普遍优于 PPO/DRL，"
        "而是说明在“9D 紧凑上下文 + 5 个离散联合动作 + 强安全约束 + 分布漂移 + 星上需在线轻量适应”"
        "这个特定 LEO 计算-通信协同设计场景里，Safe Contextual Bandit 更匹配问题结构。"
    )
    fig.text(
        0.03,
        0.975,
        title,
        fontproperties=FONT,
        fontsize=24,
        fontweight="bold",
        color="#223127",
        va="top",
    )
    fig.text(
        0.03,
        0.945,
        fill(subtitle, width=110),
        fontproperties=FONT,
        fontsize=12.5,
        color="#48555E",
        va="top",
    )

    ax_problem = fig.add_subplot(gs[0, 0])
    add_panel(ax_problem, "A. 问题设定快照")
    add_bullets(
        ax_problem,
        [
            "9D context：B, E_AI, Theta, V_prior, D, H, U, C, Q。",
            "5 个动作：lw+meta、bal+meta、bal+ROI、hi+ROI+ctx、delay/store。",
            "安全约束显式存在：battery、thermal、queue、deadline；不应依赖策略“隐式学会”。",
            "部署环境非平稳：链路、任务价值、能量补给、队列压力都会漂移。",
            "动作选择前只能看到 V_prior，不能偷看 V_realized_hidden。",
        ],
        size=9.9,
        step=0.145,
        width=31,
    )

    ax_goal = fig.add_subplot(gs[0, 1])
    add_panel(ax_goal, "B. 文档想证明什么")
    add_bullets(
        ax_goal,
        [
            "不是证明“DRL 跑不动”，而是说明它对该低维、小动作集、强约束、非平稳场景并非最佳结构匹配。",
            "Safe LinUCB 可用 observed reward 在线轻量更新；冻结式 DRL/PPO-style policy 缺少这一步。",
            "SafeFilter 让 bandit 的安全约束可解释、可审计；DRL/PPO 若不额外加 shield，容易越约束。",
            "结论边界必须收紧：这里只能说“更匹配本文设定”，不能说“普遍优于 PPO/DRL”。",
        ],
        size=9.9,
        step=0.185,
        width=32,
    )

    ax_repo = fig.add_subplot(gs[0, 2])
    add_panel(ax_repo, "C. 当前仓库快速核查")
    add_bullets(
        ax_repo,
        [
            "现有 router：fixed_lightweight、fixed_high_accuracy、utility_rule、comm_conservative、safe_linucb。",
            "已有 `outputs/summary.csv` 和基础图，但未看到对齐文档的 `planner_comparison` 产物。",
            "主要缺口：offline_drl_policy、shielded offline_drl、双环境多 seed 汇总结果。",
            "因此本图是“文档逻辑对比图”，不是完整实验结果图。",
        ],
        size=9.7,
        step=0.19,
        width=29,
    )

    ax_matrix = fig.add_subplot(gs[1, :])
    ax_matrix.set_facecolor("#FFFDF8")
    matrix_patch = FancyBboxPatch(
        (0.005, 0.02),
        0.99,
        0.96,
        boxstyle="round,pad=0.018,rounding_size=0.03",
        linewidth=1.4,
        edgecolor="#CCBFA3",
        facecolor="#FFFDF8",
        transform=ax_matrix.transAxes,
        zorder=-10,
    )
    ax_matrix.add_patch(matrix_patch)

    planners = [
        "Safe Contextual Bandit\n(Safe LinUCB)",
        "Offline DRL/PPO\n(冻结部署)",
        "Offline DRL/PPO +\nSafety Shield",
    ]
    dimensions = [
        "与 9D/5 动作\n问题结构匹配度",
        "observed reward\n在线适应能力",
        "安全性是否显式\n可过滤/可审计",
        "对 deployment shift\n鲁棒性",
        "星上推理负担\n是否轻量",
        "对重离线训练的\n依赖程度",
        "单步行为可解释性\n与调试性",
        "对队列/资源反馈\n集成自然度",
    ]
    scores = np.array(
        [
            [5, 5, 5, 4, 5, 4, 4, 5],
            [2, 1, 1, 2, 4, 1, 2, 3],
            [2, 1, 4, 2, 3, 1, 2, 3],
        ]
    )
    notes = [
        ["原生匹配", "在线更新", "Filter 内建", "可逐步修正", "矩阵小", "无需重训练", "arm 可追踪", "天然适配"],
        ["偏重方案", "冻结策略", "需额外 shield", "遇漂移脆弱", "推理可行", "依赖离线训练", "较难审计", "可做但不自然"],
        ["仍偏重", "仍冻结", "安全显著改善", "仍缺在线修正", "额外屏蔽开销", "依赖离线训练", "仍难解释", "反馈不充分"],
    ]

    cmap = colors.LinearSegmentedColormap.from_list(
        "fit_scale",
        ["#B33A3A", "#D97B2D", "#E8C867", "#89B86B", "#3E8B5B"],
    )
    ax_matrix.imshow(scores, cmap=cmap, vmin=1, vmax=5, aspect="auto", zorder=1)
    ax_matrix.set_title(
        "D. 规划器结构性对比评分（5 = 更适合当前文档定义的特定 LEO 场景）",
        fontproperties=FONT,
        fontsize=16,
        fontweight="bold",
        color="#2E3A2F",
        pad=14,
    )
    ax_matrix.set_xticks(np.arange(len(dimensions)))
    ax_matrix.set_yticks(np.arange(len(planners)))
    ax_matrix.set_xticklabels(dimensions, fontproperties=FONT, fontsize=11)
    ax_matrix.set_yticklabels(planners, fontproperties=FONT, fontsize=12)
    ax_matrix.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    for edge in ax_matrix.spines.values():
        edge.set_visible(False)
    ax_matrix.set_xticks(np.arange(-0.5, len(dimensions), 1), minor=True)
    ax_matrix.set_yticks(np.arange(-0.5, len(planners), 1), minor=True)
    ax_matrix.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax_matrix.tick_params(which="minor", bottom=False, left=False)
    for r in range(scores.shape[0]):
        for c in range(scores.shape[1]):
            ax_matrix.text(
                c,
                r,
                f"{scores[r, c]}\n{notes[r][c]}",
                ha="center",
                va="center",
                fontproperties=FONT,
                fontsize=10,
                color="#102015",
                fontweight="bold" if scores[r, c] >= 4 else "normal",
                zorder=2,
            )

    planners_short = ["Safe LinUCB", "Frozen DRL/PPO", "Frozen DRL/PPO\n+ Shield"]
    planner_colors = ["#2F7D4A", "#C75B39", "#D4A72C"]
    perf_metrics = ["总任务效用", "deadline 满足", "安全表现", "队列适应"]
    id_scores = np.array(
        [
            [5, 4, 5, 4],
            [4, 4, 2, 3],
            [4, 4, 4, 3],
        ]
    )
    shift_scores = np.array(
        [
            [5, 5, 5, 5],
            [2, 2, 2, 2],
            [3, 3, 4, 3],
        ]
    )

    ax_id = fig.add_subplot(gs[2, 0])
    y = np.arange(len(perf_metrics))
    bar_h = 0.22
    for idx, (name, color) in enumerate(zip(planners_short, planner_colors)):
        ax_id.barh(
            y + (idx - 1) * bar_h,
            id_scores[idx],
            height=bar_h,
            color=color,
            label=name,
        )
    ax_id.set_facecolor("#FFFDF8")
    ax_id.set_xlim(0, 5.2)
    ax_id.set_xticks([1, 2, 3, 4, 5])
    ax_id.set_yticks(y)
    ax_id.set_yticklabels(perf_metrics, fontproperties=FONT, fontsize=11)
    ax_id.invert_yaxis()
    ax_id.grid(axis="x", color="#E3D8C2", linewidth=0.9)
    ax_id.set_title(
        "E. 预期表现：in_distribution（定性）",
        fontproperties=FONT,
        fontsize=15,
        fontweight="bold",
        color="#2E3A2F",
        pad=10,
    )
    ax_id.tick_params(axis="x", labelsize=10)
    for edge in ax_id.spines.values():
        edge.set_visible(False)
    ax_id.legend(
        prop=FONT,
        fontsize=10,
        loc="lower right",
        frameon=False,
    )

    ax_shift = fig.add_subplot(gs[2, 1])
    for idx, color in enumerate(planner_colors):
        ax_shift.barh(
            y + (idx - 1) * bar_h,
            shift_scores[idx],
            height=bar_h,
            color=color,
        )
    ax_shift.set_facecolor("#FFFDF8")
    ax_shift.set_xlim(0, 5.2)
    ax_shift.set_xticks([1, 2, 3, 4, 5])
    ax_shift.set_yticks(y)
    ax_shift.set_yticklabels(perf_metrics, fontproperties=FONT, fontsize=11)
    ax_shift.invert_yaxis()
    ax_shift.grid(axis="x", color="#E3D8C2", linewidth=0.9)
    ax_shift.set_title(
        "F. 预期表现：shifted_orbit_environment（定性）",
        fontproperties=FONT,
        fontsize=15,
        fontweight="bold",
        color="#2E3A2F",
        pad=10,
    )
    ax_shift.tick_params(axis="x", labelsize=10)
    for edge in ax_shift.spines.values():
        edge.set_visible(False)

    ax_metrics = fig.add_subplot(gs[2, 2])
    add_panel(ax_metrics, "G. 完整实验最该看什么")
    add_bullets(
        ax_metrics,
        [
            "效用：total_mission_utility、avg_mission_utility。",
            "效率：utility_per_joule、utility_per_downlink_bit。",
            "时效：deadline_satisfaction_rate、high_value_delivery_rate、expired_high_value_tasks。",
            "安全：safety_violations、avg_safe_action_count、fallback_count。",
            "资源：total_energy、total_compute_energy、total_comm_energy、final_battery、max_thermal。",
            "通信/队列：avg_queue_pressure、max_queue_mb、generated / downlinked payload。",
            "轨迹证据：cumulative reward、action distribution、delivery_status 时间演化。",
        ],
        size=9.4,
        step=0.112,
        width=27,
    )

    fig.text(
        0.03,
        0.018,
        "注：本图依据 `粗略对比.txt` 的需求逻辑和当前仓库快速核查生成，属于“文档驱动的结构性对比图”，"
        "不是已经完成的多环境多 seed 实验结果图。",
        fontproperties=FONT,
        fontsize=11,
        color="#4E5D5F",
    )

    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
