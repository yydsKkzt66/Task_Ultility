from __future__ import annotations

from pathlib import Path
import sys
from textwrap import fill

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_project_configs

OUTPUT_DIR = ROOT / "outputs" / "doc_comparison"
OUTPUT_PATH = OUTPUT_DIR / "reward_gap_vs_rl_dashboard.png"


def pick_font() -> FontProperties:
    for name in ["Microsoft YaHei", "SimHei", "FangSong", "STFangsong", "DejaVu Sans"]:
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
        fontsize=15,
        fontweight="bold",
        color="#2E3A2F",
        va="top",
    )


def add_bullets(ax, items: list[str], start_y: float = 0.82, size: int = 10.5, step: float = 0.14, width: int = 38) -> None:
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
    cfg = load_project_configs(ROOT)
    reward_cfg = cfg["reward"]

    fig = plt.figure(figsize=(19, 12.5), facecolor="#F4F1EA")
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[0.95, 1.18, 0.98],
        width_ratios=[1.0, 1.1, 0.95],
        hspace=0.16,
        wspace=0.12,
    )

    title = "奖励函数修改后：你的方法 vs PPO / DQN / DRL 的差距重估"
    subtitle = (
        "核心判断：reward 变得更任务效用导向后，若 PPO/DQN/DRL 也用同一新 reward 重训，"
        "“目标定义差距”会缩小；但“在线适应、安全显式化、星上轻量更新”这些方法层差距仍然存在。"
    )
    fig.text(
        0.03,
        0.975,
        title,
        fontproperties=FONT,
        fontsize=23,
        fontweight="bold",
        color="#223127",
        va="top",
    )
    fig.text(
        0.03,
        0.945,
        fill(subtitle, width=110, break_long_words=False),
        fontproperties=FONT,
        fontsize=12.2,
        color="#48555E",
        va="top",
    )

    ax_formula = fig.add_subplot(gs[0, 0])
    add_panel(ax_formula, "A. 当前 reward 结构")
    formula_lines = [
        "R = V_t · Q_det · S_info · phi · rho",
        f"    - lambda_E(t)·E_norm   (lambda_E_base={reward_cfg['lambda_E_base']:.2f})",
        f"    - lambda_L·L_norm      (lambda_L={reward_cfg['lambda_L']:.2f})",
        f"    - lambda_Q·Q_norm      (lambda_Q={reward_cfg['lambda_Q']:.2f})",
        f"    - lambda_Omega·Omega   (lambda_Omega={reward_cfg['lambda_Omega']:.2f})",
        "",
        "lambda_E(t) = lambda_E_base · (1 + eta_B·(1-B_t) + eta_Theta·Theta_t)",
        f"eta_B={reward_cfg['eta_B']:.2f}, eta_Theta={reward_cfg['eta_Theta']:.2f}",
        "",
        "S_info = min(1, I(p) / I_req(x_t))",
        "I_req 受 V_prior / H / U / D 共同驱动，意味着 payload 不是越大越好。",
        "",
        "另外，代码已支持 high-value error penalty，但默认主配置未开启该开关。",
    ]
    y = 0.83
    for line in formula_lines:
        ax_formula.text(
            0.05,
            y,
            line,
            transform=ax_formula.transAxes,
            fontproperties=FONT,
            fontsize=10.3,
            color="#29323A",
            va="top",
        )
        y -= 0.07

    ax_effect = fig.add_subplot(gs[0, 1])
    add_panel(ax_effect, "B. 这次 reward 修改带来的真实影响")
    add_bullets(
        ax_effect,
        [
            "reward 不再只是“检测好坏”或简单 utility，而是把信息充分度、时效、交付状态、能耗、队列和安全约束绑在一起。",
            "因此 PPO / DQN / DRL 如果也改用这一 reward 并重新训练，它们的优化目标会更接近你的方法；这一层差距比之前更小。",
            "但 reward 越结构化、越上下文相关，越要求算法能在部署时及时利用 observed reward 修正行为；这正是 SafeLinUCB 的强项。",
            "如果你只改了自己的 reward，而 PPO / DQN / DRL 仍沿用旧 reward 或旧策略，那么比较里会额外混入 objective mismatch，不能直接视为纯算法胜负。",
        ],
        start_y=0.81,
        size=10.2,
        step=0.19,
        width=39,
    )

    ax_case = fig.add_subplot(gs[0, 2])
    add_panel(ax_case, "C. 两种比较口径")
    add_bullets(
        ax_case,
        [
            "口径 1：所有方法共享你现在这版 reward，并且 PPO / DQN / DRL 全部重训。此时主要看方法本身和部署方式差异。",
            "口径 2：只有你的方法按新 reward 更新，而强化学习基线没重训。此时差距会被放大，其中一部分来自 reward 定义不一致。",
            "如果你的目的是论文或答辩中的公平对比，应该优先使用口径 1；口径 2 只能当“reward 改动敏感性”说明。",
        ],
        start_y=0.81,
        size=10.0,
        step=0.23,
        width=32,
    )

    ax_matrix = fig.add_subplot(gs[1, :])
    rows = ["你的方法\nSafeLinUCB", "DQN", "PPO", "典型冻结式\nDRL"]
    cols = [
        "当前复合 reward\n结构匹配",
        "observed reward\n在线闭环",
        "5 离散动作\n适配度",
        "显式安全过滤\n集成度",
        "奖励修改后\n免重训程度",
        "星上部署\n轻量性",
        "非平稳环境\n鲁棒性",
    ]
    scores = np.array(
        [
            [5, 5, 5, 5, 5, 5, 5],
            [4, 3, 5, 2, 2, 2, 3],
            [4, 2, 3, 2, 2, 2, 2],
            [4, 1, 3, 2, 1, 4, 2],
        ]
    )
    notes = [
        ["原生一跳", "原生更新", "动作正好小", "Filter 外挂自然", "无需重训", "矩阵最轻", "适应较强"],
        ["可学到", "可做但更重", "离散友好", "需额外 mask", "通常要重训", "若在线更重", "中等"],
        ["可学到", "在线代价高", "可做但不优", "需额外 mask", "通常要重训", "在线更重", "偏弱"],
        ["目标接近", "冻结无更新", "可推理", "需额外 shield", "必须重训", "推理可行", "遇漂移弱"],
    ]
    cmap = colors.LinearSegmentedColormap.from_list(
        "gap_scale",
        ["#B73D3D", "#DB812F", "#ECD36E", "#8FBF69", "#3F8B5C"],
    )
    ax_matrix.imshow(scores, cmap=cmap, vmin=1, vmax=5, aspect="auto")
    ax_matrix.set_title(
        "D. 若围绕当前 reward 比较方法本身，谁更匹配你的场景（5 = 更匹配，定性评分）",
        fontproperties=FONT,
        fontsize=16,
        fontweight="bold",
        color="#2E3A2F",
        pad=14,
    )
    ax_matrix.set_xticks(np.arange(len(cols)))
    ax_matrix.set_yticks(np.arange(len(rows)))
    ax_matrix.set_xticklabels(cols, fontproperties=FONT, fontsize=11)
    ax_matrix.set_yticklabels(rows, fontproperties=FONT, fontsize=12)
    ax_matrix.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, length=0)
    for edge in ax_matrix.spines.values():
        edge.set_visible(False)
    ax_matrix.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax_matrix.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
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
                fontsize=9.8,
                color="#102015",
                fontweight="bold" if scores[r, c] >= 4 else "normal",
            )

    ax_gap = fig.add_subplot(gs[2, 0:2])
    methods = ["DQN", "PPO", "冻结式 DRL"]
    shared_reward_gap = np.array([2.6, 3.1, 4.1])
    unretrained_gap = np.array([4.0, 4.4, 4.8])
    x = np.arange(len(methods))
    w = 0.32
    ax_gap.bar(x - w / 2, shared_reward_gap, width=w, color="#3F8B5C", label="共用新 reward 并重训")
    ax_gap.bar(x + w / 2, unretrained_gap, width=w, color="#C75B39", label="RL 未按新 reward 重训")
    ax_gap.set_facecolor("#FFFDF8")
    ax_gap.set_ylim(0, 5.2)
    ax_gap.set_yticks([1, 2, 3, 4, 5])
    ax_gap.set_xticks(x)
    ax_gap.set_xticklabels(methods, fontproperties=FONT, fontsize=11)
    ax_gap.grid(axis="y", color="#E3D8C2", linewidth=0.9)
    ax_gap.set_title(
        "E. 你和各类 RL 的“综合差距”会怎么变（定性）",
        fontproperties=FONT,
        fontsize=15,
        fontweight="bold",
        color="#2E3A2F",
        pad=10,
    )
    ax_gap.legend(prop=FONT, frameon=False, fontsize=10, loc="upper left")
    for edge in ax_gap.spines.values():
        edge.set_visible(False)
    ax_gap.text(
        0.02,
        0.02,
        "解释：reward 统一并重训后，差距主要来自在线适应、安全显式化和部署轻量性；\n"
        "若基线未重训，则额外叠加 objective mismatch，比较会更偏向你的方法。",
        transform=ax_gap.transAxes,
        fontproperties=FONT,
        fontsize=10,
        color="#4A5960",
        va="bottom",
    )

    ax_take = fig.add_subplot(gs[2, 2])
    add_panel(ax_take, "F. 结论怎么说更严谨")
    add_bullets(
        ax_take,
        [
            "你现在的 reward 更适合拿来说明“任务效用驱动 co-design”，而不是单一检测指标优化。",
            "和 PPO / DQN / DRL 的差距，不应再简单表述为“reward 不一样”；更准确是“即便 reward 对齐，方法与部署方式仍不一样”。",
            "DQN 比 PPO 更接近你这个 5 离散动作场景，但在显式安全与轻量在线更新上仍不如 SafeLinUCB。",
            "PPO / 泛化 DRL 的优势是更强的长时域策略表达；你的场景当前更像小动作集、强约束、一步效用主导的问题，因此 bandit 更顺手。",
        ],
        start_y=0.81,
        size=10.0,
        step=0.19,
        width=31,
    )

    fig.text(
        0.03,
        0.02,
        "注：图中“差距评分”是基于当前代码结构与部署假设的定性判断，不是已经完成 DQN/PPO 实验后的实测数值。",
        fontproperties=FONT,
        fontsize=10.8,
        color="#4E5D5F",
    )

    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
