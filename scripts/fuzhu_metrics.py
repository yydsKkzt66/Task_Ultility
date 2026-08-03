from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

# ============================================================
# 1. Output directory
# ============================================================
OUTPUT_DIR = Path("auxiliary_metric_figures_v4")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. Methods and fixed colors
# ============================================================
methods = [
    "Myopic",
    "B1",
    "OFFLOAD",
    "TUT-v2",
    "PROP",
]

method_labels_singleline = [
    "Myopic",
    "B1",
    "OFFLOAD",
    "TUT-v2",
    "PROP",
]

colors = [
    "#B2E6FD",  # Myopic + REQALIGN
    "#B8D2CC",  # B1
    "#E8B2A7",  # OFFLOAD
    "#FEEBB9",  # TUT-v2
    "#B5AED5",  # Corrected REQALIGN
]

proposed_index = 4
proposed_color = colors[proposed_index]

# ============================================================
# 3. Data
# ============================================================
hv_metadata = np.array([37.0, 0.0, 0.0, 4.0, 45.0])
hv_roi      = np.array([ 4.6, 0.0, 3.2, 28.8,  6.6])
hv_evidence = np.array([ 4.0,44.8,40.8, 22.0, 37.8])
hv_total = hv_metadata + hv_roi + hv_evidence

hv_latency_mean = np.array([2584, 2643, 2819, 1979, 2354])
hv_latency_p95  = np.array([4276, 4694, 4538, 4284, 4500])

hv_net_utility = np.array([8.645, 57.984, 46.393, 48.596, 60.444])
energy_wh      = np.array([1.825, 4.177, 6.074, 2.421, 2.210])
downlink_mb    = np.array([119.857, 676.000, 487.650, 142.111, 184.694])

# ============================================================
# 4. Font settings
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "hatch.linewidth": 1.2,
})

available_fonts = {font.name for font in font_manager.fontManager.ttflist}
if "Times New Roman" not in available_fonts:
    print("Warning: Times New Roman not found. Using fallback serif font.")

TITLE_SIZE = 22
AXIS_LABEL_SIZE = 20

# 数字
TICK_SIZE = 26
VALUE_SIZE = 22

# 图2纵轴数字缩小20%
P2_YTICK_SIZE = int(TICK_SIZE * 0.8)  # 20.8 -> 20

# 方法名：在上一版基础上再增大20%
METHOD_SIZE = 24

# 图例和图注
LEGEND_SIZE = 24
NOTE_SIZE = 22
ANNOT_SIZE = 18

# ============================================================
# 5. Figure canvas
# ============================================================
fig, axes = plt.subplots(
    1, 3,
    figsize=(26, 9.5),
    gridspec_kw={"width_ratios": [1.15, 1.12, 1.05]}
)
ax1, ax2, ax3 = axes

# ============================================================
# 6. Panel (a): stacked bar chart
# ============================================================
x = np.arange(len(methods))
width = 0.64

for i in range(len(methods)):
    edge_lw = 2.8 if i == proposed_index else 1.5

    ax1.bar(
        x[i], hv_metadata[i], width=width,
        color=colors[i], edgecolor="#666666", linewidth=edge_lw,
        alpha=0.35, hatch="////", zorder=3
    )
    ax1.bar(
        x[i], hv_roi[i], width=width, bottom=hv_metadata[i],
        color=colors[i], edgecolor="#666666", linewidth=edge_lw,
        alpha=0.65, hatch="xx", zorder=3
    )
    ax1.bar(
        x[i], hv_evidence[i], width=width, bottom=hv_metadata[i] + hv_roi[i],
        color=colors[i], edgecolor="#666666", linewidth=edge_lw,
        alpha=0.95, zorder=3
    )

for i in range(len(methods)):
    if hv_metadata[i] >= 3:
        ax1.text(
            x[i], hv_metadata[i] / 2, f"{hv_metadata[i]:.1f}",
            ha="center", va="center", fontsize=VALUE_SIZE
        )
    if hv_roi[i] >= 3:
        ax1.text(
            x[i], hv_metadata[i] + hv_roi[i] / 2, f"{hv_roi[i]:.1f}",
            ha="center", va="center", fontsize=VALUE_SIZE
        )
    if hv_evidence[i] >= 3:
        ax1.text(
            x[i], hv_metadata[i] + hv_roi[i] + hv_evidence[i] / 2,
            f"{hv_evidence[i]:.1f}",
            ha="center", va="center", fontsize=VALUE_SIZE
        )

# 柱顶总值
for i in range(len(methods)):
    ax1.text(
        x[i], hv_total[i] + 2.3, f"{hv_total[i]:.1f}",
        ha="center", va="bottom",
        fontsize=VALUE_SIZE,
        fontweight="bold"
    )

ax1.set_title("(a) High-Value Layer Deliveries", fontsize=TITLE_SIZE, pad=14)
ax1.set_ylabel("Delivered Layer Count", fontsize=AXIS_LABEL_SIZE)
ax1.set_xticks(x)
ax1.set_xticklabels(methods, rotation=35, ha="right", fontsize=METHOD_SIZE)
ax1.set_ylim(0, 100)
ax1.yaxis.set_major_locator(MultipleLocator(20))
ax1.yaxis.set_minor_locator(MultipleLocator(10))
ax1.tick_params(axis="y", labelsize=TICK_SIZE)
ax1.tick_params(axis="x", labelsize=METHOD_SIZE)

layer_legend = [
    Patch(facecolor="white", edgecolor="#666666", hatch="////", label="Metadata"),
    Patch(facecolor="white", edgecolor="#666666", hatch="xx", label="ROI"),
    Patch(facecolor="#C9C9C9", edgecolor="#666666", label="Evidence"),
]
ax1.legend(
    handles=layer_legend,
    loc="upper left",
    frameon=False,
    fontsize=LEGEND_SIZE,
    handlelength=1.6
)

# ============================================================
# 7. Panel (b): vertical dumbbell plot
# ============================================================
x2 = np.arange(len(methods))

for i in range(len(methods)):
    lw = 4.2 if i == proposed_index else 3.0
    ms_mean = 220 if i == proposed_index else 180
    ms_p95 = 210 if i == proposed_index else 170

    ax2.plot(
        [x2[i], x2[i]],
        [hv_latency_mean[i], hv_latency_p95[i]],
        color=colors[i],
        linewidth=lw,
        zorder=2
    )

    ax2.scatter(
        x2[i], hv_latency_mean[i],
        s=ms_mean, marker="o", color=colors[i],
        edgecolor="#444444", linewidth=2.0, zorder=4
    )

    ax2.scatter(
        x2[i], hv_latency_p95[i],
        s=ms_p95, marker="D", color=colors[i],
        edgecolor="#444444", linewidth=2.0, zorder=4
    )

    ax2.text(
        x2[i], hv_latency_mean[i] - 130,
        f"{hv_latency_mean[i]:.0f}",
        ha="center", va="top", fontsize=VALUE_SIZE
    )
    ax2.text(
        x2[i], hv_latency_p95[i] + 130,
        f"{hv_latency_p95[i]:.0f}",
        ha="center", va="bottom", fontsize=VALUE_SIZE
    )

# 紫色虚线
ax2.hlines(
    y=hv_latency_mean[proposed_index],
    xmin=-0.25,
    xmax=x2[proposed_index],
    colors=proposed_color,
    linestyles="--",
    linewidth=2.8,
    alpha=0.95,
    zorder=1
)

ax2.set_title("(b) High-Value Closure Latency", fontsize=TITLE_SIZE, pad=14)
ax2.set_ylabel("Ground Closure Latency (s)", fontsize=AXIS_LABEL_SIZE)
ax2.set_xticks(x2)
ax2.set_xticklabels(methods, rotation=35, ha="right", fontsize=METHOD_SIZE)
ax2.set_ylim(1500, 5100)
ax2.set_xlim(-0.25, len(methods) - 0.75)
ax2.yaxis.set_major_locator(MultipleLocator(500))
ax2.yaxis.set_minor_locator(MultipleLocator(250))

# 图2：纵轴数字缩小20%
ax2.tick_params(axis="y", labelsize=P2_YTICK_SIZE)
ax2.tick_params(axis="x", labelsize=METHOD_SIZE)

latency_legend = [
    Line2D([0], [0], marker='o', color='none',
           markerfacecolor="#BBBBBB", markeredgecolor="#444444",
           markersize=12, label="Mean"),
    Line2D([0], [0], marker='D', color='none',
           markerfacecolor="#BBBBBB", markeredgecolor="#444444",
           markersize=12, label="P95"),
]
ax2.legend(
    handles=latency_legend,
    loc="upper left",
bbox_to_anchor=(0.02, 0.22),  # 这里控制具体位置
    frameon=False,
    fontsize=LEGEND_SIZE,
    handletextpad=0.2,     # 文字更靠近图形
    borderaxespad=0.2,
    labelspacing=0.4
)

# ============================================================
# 8. Panel (c): utility–resource tradeoff
# ============================================================

# ------------------------------------------------------------
# Bubble area is proportional to downlink volume
# Matplotlib parameter s represents marker area in points^2
# ------------------------------------------------------------
bubble_scale = 2.6
bubble_sizes = downlink_mb * bubble_scale

# Draw bubbles
for i in range(len(methods)):
    edge_lw = 3.2 if i == proposed_index else 2.0

    ax3.scatter(
        energy_wh[i],
        hv_net_utility[i],
        s=bubble_sizes[i],
        color=colors[i],
        edgecolor="#444444",
        linewidth=edge_lw,
        alpha=0.88,
        zorder=4,
    )

# ------------------------------------------------------------
# Put every method name outside the lower-right edge
# of its corresponding bubble
# ------------------------------------------------------------
for i in range(len(methods)):

    # Convert bubble area to an approximate radius in points,
    # then use it to place each label outside the bubble.
    bubble_radius = np.sqrt(bubble_sizes[i] / np.pi)

    offset_x = 0.80 * bubble_radius + 4
    offset_y = -(0.80 * bubble_radius + 4)

    ax3.annotate(
        method_labels_singleline[i],
        xy=(energy_wh[i], hv_net_utility[i]),
        xytext=(offset_x, offset_y),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=ANNOT_SIZE,
        color="black",
        zorder=10,
        annotation_clip=False,
    )

# ------------------------------------------------------------
# Title and axes
# ------------------------------------------------------------
ax3.set_title(
    "(c) Utility–Resource Tradeoff",
    fontsize=TITLE_SIZE,
    pad=14,
)

# Retain your 20% larger x-axis title
X_AXIS_LABEL_SIZE = AXIS_LABEL_SIZE * 1.20

ax3.set_xlabel(
    "Controllable Energy (Wh)",
    fontsize=X_AXIS_LABEL_SIZE,
    labelpad=12,
)

ax3.set_ylabel(
    r"High-Value Net Utility $J_{\mathrm{HV}}^{\mathrm{net}}$",
    fontsize=AXIS_LABEL_SIZE,
    labelpad=12,
)

# Retain the extra right margin for OFFLOAD and its label
ax3.set_xlim(1.1, 7.25)
ax3.set_ylim(0, 70)

ax3.xaxis.set_major_locator(MultipleLocator(1))
ax3.xaxis.set_minor_locator(MultipleLocator(0.5))
ax3.yaxis.set_major_locator(MultipleLocator(10))
ax3.yaxis.set_minor_locator(MultipleLocator(5))

ax3.tick_params(
    axis="x",
    which="major",
    labelsize=TICK_SIZE,
    width=1.8,
    length=6,
)

ax3.tick_params(
    axis="y",
    which="major",
    labelsize=TICK_SIZE,
    width=1.8,
    length=6,
)

ax3.tick_params(
    axis="both",
    which="minor",
    width=1.2,
    length=3.5,
)

# ------------------------------------------------------------
# Bubble-size legend: downlink volume
# Three reference bubbles make the area encoding interpretable
# ------------------------------------------------------------
reference_volumes = [150, 400, 650]

bubble_legend_handles = [
    ax3.scatter(
        [],
        [],
        s=volume * bubble_scale,
        facecolor="none",
        edgecolor="#555555",
        linewidth=1.4,
        label=f"{volume} MB",
    )
    for volume in reference_volumes
]

bubble_legend = ax3.legend(
    handles=bubble_legend_handles,
    title="Downlink volume",

    # 整体位置：保持右下角，并适当下移
    loc="lower right",
    bbox_to_anchor=(0.98, 0.025),

    frameon=False,

    # 字体
    fontsize=19,
    title_fontsize=21,

    scatterpoints=1,

    # 压缩三个气泡之间的纵向距离
    labelspacing=1.35,

    # 缩短气泡与文字之间的距离
    handletextpad=0.75,

    # 减少图例内部留白
    borderpad=0.10,
    borderaxespad=0.0,

    # 保持文字左对齐
    alignment="left",
)

# Keep the legend above other visual elements
bubble_legend.set_zorder(20)

# ------------------------------------------------------------
# Clean journal-style frame
# ------------------------------------------------------------
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.spines["left"].set_linewidth(1.8)
ax3.spines["bottom"].set_linewidth(1.8)

ax3.grid(False)
# ============================================================
# 9. Common styling
# ============================================================
for ax in (ax1, ax2, ax3):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_linewidth(1.8)

    ax.grid(False)
# ============================================================
# 10. Figure note
# ============================================================
fig.text(
    0.03,
    0.035,
    (
        "Note: Panel (a) reports non-exclusive layer-delivery counts. "
        "A task may contribute to multiple layers; therefore, the stacked "
        "total represents delivery events rather than unique tasks."
    ),
    ha="left",
    va="bottom",
    fontsize=NOTE_SIZE,
)

# ============================================================
# 11. Layout
# ============================================================
fig.subplots_adjust(
    left=0.065,
    right=0.985,
    top=0.87,
    bottom=0.27,
    wspace=0.38,
)

# ============================================================
# 12. Save and display
# ============================================================
pdf_path = OUTPUT_DIR / "three_auxiliary_metrics_refined.pdf"
png_path = OUTPUT_DIR / "three_auxiliary_metrics_refined.png"

fig.savefig(
    pdf_path,
    bbox_inches="tight",
    facecolor="white",
)

fig.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

print(f"Saved PDF to: {pdf_path.resolve()}")
print(f"Saved PNG to: {png_path.resolve()}")

plt.show()