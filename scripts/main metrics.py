from pathlib import Path
import csv
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# ============================================================
# 1. Output settings
# ============================================================
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "artifacts" / "competition_release_v1" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Methods and fixed colors
#    Keep exactly the same method-color order as the old figure
# ============================================================
methods = [
    "Myopic",
    "B1",
    "OFFLOAD",
    "TUT",
    "PROP",
]

method_labels_singleline = methods.copy()

colors = [
    "#B2E6FD",  # Myopic
    "#B8D2CC",  # B1
    "#E8B2A7",  # OFFLOAD
    "#FEEBB9",  # TUT
    "#B5AED5",  # PROP
]

proposed_index = 4
proposed_color = colors[proposed_index]

x = np.arange(len(methods))


# ============================================================
# 3. Automatic axis helpers
# ============================================================
def nice_number(value: float, rounding: bool = False) -> float:
    """Return a visually pleasant tick interval close to value."""
    if value == 0:
        return 1.0

    exponent = math.floor(math.log10(abs(value)))
    fraction = abs(value) / (10 ** exponent)

    if rounding:
        if fraction < 1.5:
            nice_fraction = 1
        elif fraction < 2.25:
            nice_fraction = 2
        elif fraction < 3.75:
            nice_fraction = 2.5
        elif fraction < 7.5:
            nice_fraction = 5
        else:
            nice_fraction = 10
    else:
        if fraction <= 1:
            nice_fraction = 1
        elif fraction <= 2:
            nice_fraction = 2
        elif fraction <= 2.5:
            nice_fraction = 2.5
        elif fraction <= 5:
            nice_fraction = 5
        else:
            nice_fraction = 10

    return nice_fraction * (10 ** exponent)


def compute_axis_limits(
    values,
    errors=None,
    include_zero=True,
    n_ticks=6,
    pad_ratio=0.12,
    force_zero_bottom=False,
):
    """Compute automatic limits and a pleasant major-tick interval."""
    values = np.asarray(values, dtype=float)

    if errors is None:
        data_min = float(np.min(values))
        data_max = float(np.max(values))
    else:
        errors = np.asarray(errors, dtype=float)
        data_min = float(np.min(values - errors))
        data_max = float(np.max(values + errors))

    if force_zero_bottom:
        data_min = 0.0
        data_max = max(data_max, 0.0)
    elif include_zero:
        data_min = min(data_min, 0.0)
        data_max = max(data_max, 0.0)

    if np.isclose(data_min, data_max):
        if np.isclose(data_min, 0.0):
            data_min, data_max = 0.0, 1.0
        else:
            delta = abs(data_min) * 0.20
            data_min -= delta
            data_max += delta

    span = data_max - data_min
    pad = span * pad_ratio

    raw_min = data_min if force_zero_bottom else data_min - pad
    raw_max = data_max + pad

    raw_step = (raw_max - raw_min) / max(n_ticks - 1, 1)
    major_step = nice_number(raw_step, rounding=True)
    major_step = max(abs(major_step), 1e-12)

    y_min = 0.0 if force_zero_bottom else math.floor(raw_min / major_step) * major_step
    y_max = math.ceil(raw_max / major_step) * major_step

    if abs(y_min) < 1e-12:
        y_min = 0.0
    if abs(y_max) < 1e-12:
        y_max = 0.0

    return y_min, y_max, major_step


def add_bar_value_label(ax, x_position, value, error, decimals=2):
    """Place labels correctly for positive and negative bars."""
    y_min, y_max = ax.get_ylim()
    span = y_max - y_min
    offset = 0.020 * span

    if value >= 0:
        y_text = value + error + offset
        va = "bottom"
    else:
        y_text = value - error - offset
        va = "top"

    ax.text(
        x_position,
        y_text,
        f"{value:.{decimals}f}",
        ha="center",
        va=va,
        fontsize=VALUE_LABEL_SIZE,
        color="black",
    )


# ============================================================
# 4. Frozen five-seed main-experiment data
#    Read the compact, Git-tracked simulation summary instead of duplicating
#    publication numbers in this plotting script.
# ============================================================
RESULT_TABLE = (
    ROOT
    / "outputs"
    / "_current_prop_four_baselines_fiveseed_dual_thermal_v1"
    / "attempt_001"
    / "tables"
    / "mean_std_metrics.csv"
)
METHOD_IDS = [
    "MYOPIC-E2E-UTILITY-GREEDY-STANDALONE-v1",
    "B1-COMMON-GROUND-THERMAL-CORRECTED",
    "OFFLOAD-EFT-THERMAL-ALIGNED-2NODE-DIAGNOSTIC",
    "TUT-PROXY-THERMAL-CORRECTED",
    "PROP-V2-SAFE-METADATA-CURRENT-v1",
]

with RESULT_TABLE.open("r", encoding="utf-8-sig", newline="") as stream:
    rows_by_method = {row["method"]: row for row in csv.DictReader(stream)}

missing_methods = [method for method in METHOD_IDS if method not in rows_by_method]
if missing_methods:
    raise RuntimeError(f"Missing methods in {RESULT_TABLE}: {missing_methods}")
if any(int(rows_by_method[method]["seed_count"]) != 5 for method in METHOD_IDS):
    raise RuntimeError(f"Expected five seeds per method in {RESULT_TABLE}")


def metric_array(field: str) -> np.ndarray:
    return np.array([float(rows_by_method[method][field]) for method in METHOD_IDS])


primary_metrics = [
    {
        "panel": "(a)",
        "title": "Overall Net Utility",
        "ylabel": r"Overall Task Net Utility $J_{\mathrm{all}}^{\mathrm{net}}$",
        "means": metric_array("J_all_net_mean"),
        "stds": metric_array("J_all_net_std"),
        "value_decimals": 2,
        "force_zero_bottom": False,
    },
    {
        "panel": "(b)",
        "title": "High-Value Net Utility",
        "ylabel": r"High-Value Net Utility $J_{\mathrm{HV}}^{\mathrm{net}}$",
        "means": metric_array("J_HV_net_mean"),
        "stds": metric_array("J_HV_net_std"),
        "value_decimals": 2,
        "force_zero_bottom": False,
    },
    {
        "panel": "(c)",
        "title": "High-Value Task Closures",
        "ylabel": r"High-Value Task Closures $N_{\mathrm{HV}}^{\mathrm{closure}}$",
        "means": metric_array("HV_closure_count_mean"),
        "stds": metric_array("HV_closure_count_std"),
        "value_decimals": 1,
        # Requested: panel (c) should sit on the horizontal axis.
        "force_zero_bottom": True,
    },
]


# Panel (d): non-exclusive HV layer-delivery counts
hv_metadata = metric_array("HV_metadata_delivered_count_mean")
hv_roi = metric_array("HV_ROI_delivered_count_mean")
hv_evidence = metric_array("HV_evidence_delivered_count_mean")
hv_total = hv_metadata + hv_roi + hv_evidence


# Panel (e): overall closure latency, seconds
all_latency_mean = metric_array("mean_closure_latency_s_mean")
all_latency_p95 = metric_array("p95_closure_latency_s_mean")


# Panel (f): utility-resource-downlink tradeoff
hv_net_utility = metric_array("J_HV_net_mean")
energy_wh = metric_array("controllable_energy_Wh_mean")
downlink_mb = metric_array("transmitted_MB_mean")


# ============================================================
# 5. Publication font settings
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 2.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "hatch.linewidth": 1.4,
})

available_fonts = {font.name for font in font_manager.fontManager.ttflist}
if "Times New Roman" not in available_fonts:
    print(
        "Warning: Times New Roman was not found. "
        "Matplotlib will use the configured serif fallback."
    )


# ============================================================
# 6. Font sizes
# ============================================================
FONT_SCALE = 1.30

ROW_TITLE_SIZE = round(28 * FONT_SCALE)
PANEL_TITLE_SIZE = round(22 * FONT_SCALE * 1.25)  # six panel titles: 1.25x
PANEL_INDEX_SIZE = round(22 * FONT_SCALE)
AXIS_LABEL_SIZE = round(19 * FONT_SCALE)
X_AXIS_LABEL_SIZE = round(21 * FONT_SCALE)
METHOD_LABEL_SIZE = round(17 * FONT_SCALE * 1.35)
TICK_NUMBER_SIZE = round(17 * FONT_SCALE * 1.50)
VALUE_LABEL_SIZE = round(15 * FONT_SCALE * 1.50)
LEGEND_SIZE = round(14 * FONT_SCALE)
ANNOT_SIZE = round(14 * FONT_SCALE)
BUBBLE_LEGEND_SIZE = round(13 * FONT_SCALE)
BUBBLE_TITLE_SIZE = round(14 * FONT_SCALE)

# Original sizes retained as bases.
LAYER_LEGEND_SIZE = round(LEGEND_SIZE * 2.0)
LATENCY_LEGEND_SIZE = round(LEGEND_SIZE * 2.0)

# Requested legend reductions.
LAYER_LEGEND_SIZE_MODIFIED = max(1, round(LAYER_LEGEND_SIZE * 0.75))
LATENCY_LEGEND_SIZE_MODIFIED = max(1, round(LATENCY_LEGEND_SIZE * 0.80))

# Requested: double the bubble-legend numeric-label size.
BUBBLE_LEGEND_NUMBER_SIZE = max(1, round(BUBBLE_LEGEND_SIZE * 2.0))

# Unified legend settings
D_LEGEND_FONT_SIZE = LAYER_LEGEND_SIZE_MODIFIED
E_LEGEND_FONT_SIZE = LATENCY_LEGEND_SIZE_MODIFIED
D_LEGEND_HANDLE_TEXT_PAD = 0.35

LEGEND_FRAME_EDGE_COLOR = "black"
LEGEND_FRAME_FACE_COLOR = "white"
LEGEND_FRAME_LINEWIDTH = 1.6
LEGEND_FRAME_ALPHA = 0.96


# ============================================================
# 7. Figure and 2x2 GridSpec
# ============================================================
fig = plt.figure(figsize=(20, 17), facecolor="white")

grid = fig.add_gridspec(
    nrows=2,
    ncols=2,
    hspace=0.52,
    wspace=0.30,
)


# ============================================================
# 8. Create plotting axes: (a)(b) above, (c)(d) below
# ============================================================
primary_axes = [
    fig.add_subplot(grid[0, 0]),
    fig.add_subplot(grid[0, 1]),
    fig.add_subplot(grid[1, 0]),
]
ax_f = fig.add_subplot(grid[1, 1])

all_plot_axes = [*primary_axes, ax_f]


# ============================================================
# 9. Panels (a), (b), and (c): primary metrics
# ============================================================
for ax, metric in zip(primary_axes, primary_metrics):
    means = metric["means"]
    stds = metric["stds"]

    bars = ax.bar(
        x=x,
        height=means,
        width=0.64,
        color=colors,
        edgecolor=colors,
        linewidth=2.1,
        alpha=0.90,
        yerr=stds,
        capsize=7,
        error_kw={
            "elinewidth": 2.1,
            "capthick": 2.1,
            "ecolor": "#555555",
        },
        zorder=3,
    )

    x_left = -0.55
    x_right = len(methods) - 0.45
    ax.set_xlim(x_left, x_right)

    prop_x_fraction = (
        (x[proposed_index] - x_left)
        / (x_right - x_left)
    )

    ax.axhline(
        y=means[proposed_index],
        xmin=0.0,
        xmax=prop_x_fraction,
        color=proposed_color,
        linestyle="--",
        linewidth=2.8,
        alpha=0.92,
        zorder=1,
        dash_capstyle="butt",
    )

    y_min, y_max, major_step = compute_axis_limits(
        values=means,
        errors=stds,
        include_zero=True,
        n_ticks=6,
        pad_ratio=0.14,
        force_zero_bottom=metric["force_zero_bottom"],
    )

    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_locator(MultipleLocator(major_step))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    for bar, mean, std in zip(bars, means, stds):
        x_center = bar.get_x() + bar.get_width() / 2
        add_bar_value_label(
            ax=ax,
            x_position=x_center,
            value=mean,
            error=std,
            decimals=metric["value_decimals"],
        )

    ax.set_title(
        metric["title"],
        fontsize=PANEL_TITLE_SIZE,
        pad=14,
    )

    ax.set_ylabel(
        metric["ylabel"],
        fontsize=AXIS_LABEL_SIZE,
        labelpad=12,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        methods,
        rotation=30,
        ha="right",
        rotation_mode="anchor",
        fontsize=METHOD_LABEL_SIZE,
    )

    ax.tick_params(
        axis="y",
        which="major",
        labelsize=TICK_NUMBER_SIZE,
        width=1.8,
        length=7,
    )
    ax.tick_params(
        axis="y",
        which="minor",
        width=1.3,
        length=4,
    )
    ax.tick_params(
        axis="x",
        which="major",
        labelsize=METHOD_LABEL_SIZE,
        width=1.8,
        length=7,
        pad=7,
    )


# ============================================================
# 11. Panel (d): HV layer deliveries
# ============================================================
"""Legacy panels (d) and (e) intentionally omitted from the 2x2 figure.
bar_width = 0.64

for i in range(len(methods)):
    edge_lw = 2.6 if i == proposed_index else 1.5

    ax_d.bar(
        x[i],
        hv_metadata[i],
        width=bar_width,
        color=colors[i],
        edgecolor="#666666",
        linewidth=edge_lw,
        alpha=0.35,
        hatch="////",
        zorder=3,
    )

    ax_d.bar(
        x[i],
        hv_roi[i],
        width=bar_width,
        bottom=hv_metadata[i],
        color=colors[i],
        edgecolor="#666666",
        linewidth=edge_lw,
        alpha=0.65,
        hatch="xx",
        zorder=3,
    )

    ax_d.bar(
        x[i],
        hv_evidence[i],
        width=bar_width,
        bottom=hv_metadata[i] + hv_roi[i],
        color=colors[i],
        edgecolor="#666666",
        linewidth=edge_lw,
        alpha=0.95,
        zorder=3,
    )


# Automatic upper limit, but requested lower limit is exactly zero.
ymin_d, ymax_d, major_step_d = compute_axis_limits(
    hv_total,
    include_zero=True,
    n_ticks=6,
    pad_ratio=0.18,
    force_zero_bottom=True,
)
ax_d.set_ylim(0.0, ymax_d)

for i in range(len(methods)):
    if hv_metadata[i] >= 3:
        ax_d.text(
            x[i],
            hv_metadata[i] / 2,
            f"{hv_metadata[i]:.1f}",
            ha="center",
            va="center",
            fontsize=VALUE_LABEL_SIZE,
        )

    if hv_roi[i] >= 3:
        ax_d.text(
            x[i],
            hv_metadata[i] + hv_roi[i] / 2,
            f"{hv_roi[i]:.1f}",
            ha="center",
            va="center",
            fontsize=VALUE_LABEL_SIZE,
        )

    if hv_evidence[i] >= 3:
        ax_d.text(
            x[i],
            hv_metadata[i] + hv_roi[i] + hv_evidence[i] / 2,
            f"{hv_evidence[i]:.1f}",
            ha="center",
            va="center",
            fontsize=VALUE_LABEL_SIZE,
        )

    ax_d.text(
        x[i],
        hv_total[i] + 0.020 * ymax_d,
        f"{hv_total[i]:.1f}",
        ha="center",
        va="bottom",
        fontsize=VALUE_LABEL_SIZE,
        fontweight="bold",
    )

ax_d.set_title(
    "(d) High-Value Layer Deliveries",
    fontsize=PANEL_TITLE_SIZE,
    pad=14,
)

ax_d.set_ylabel(
    "Delivered Layer Count",
    fontsize=AXIS_LABEL_SIZE,
    labelpad=12,
)

ax_d.set_xticks(x)
ax_d.set_xticklabels(
    methods,
    rotation=30,
    ha="right",
    fontsize=METHOD_LABEL_SIZE,
)

ax_d.set_xlim(-0.55, len(methods) - 0.45)
ax_d.yaxis.set_major_locator(MultipleLocator(major_step_d))
ax_d.yaxis.set_minor_locator(AutoMinorLocator(2))

ax_d.tick_params(
    axis="both",
    which="major",
    labelsize=TICK_NUMBER_SIZE,
    width=1.8,
    length=7,
)

layer_legend = [
    Patch(
        facecolor="white",
        edgecolor="#666666",
        hatch="////",
        label="Metadata",
    ),
    Patch(
        facecolor="white",
        edgecolor="#666666",
        hatch="xx",
        label="ROI",
    ),
    Patch(
        facecolor="#C9C9C9",
        edgecolor="#666666",
        label="Evidence",
    ),
]

# Requested:
# 1) legend font reduced by 25%;
# 2) legend moved slightly upward to avoid covering the first bar.
legend_d = ax_d.legend(
    handles=layer_legend,

    # 图例整体位置
    loc="upper left",
    bbox_to_anchor=(0.015, 0.985),

    # 打开图例边框
    frameon=True,

    # 图例文字大小
    fontsize=D_LEGEND_FONT_SIZE,

    # 图例符号宽度
    handlelength=1.30,

    # 图例符号和文字之间的水平间距
    handletextpad=D_LEGEND_HANDLE_TEXT_PAD,

    # Metadata、ROI、Evidence三行之间的纵向距离
    labelspacing=0.35,

    # 图例内容与边框之间的内部留白
    borderpad=0.35,

    # 图例与坐标轴边界之间的距离
    borderaxespad=0.10,
)

# 图例边框样式
legend_d.get_frame().set_edgecolor(
    LEGEND_FRAME_EDGE_COLOR
)
legend_d.get_frame().set_linewidth(
    LEGEND_FRAME_LINEWIDTH
)
legend_d.get_frame().set_facecolor(
    LEGEND_FRAME_FACE_COLOR
)
legend_d.get_frame().set_alpha(
    LEGEND_FRAME_ALPHA
)

legend_d.set_zorder(20)


# ============================================================
# 12. Panel (e): Overall closure latency
# ============================================================
x_e = np.arange(len(methods))

for i in range(len(methods)):
    line_width = 3.8 if i == proposed_index else 2.7
    mean_size = 205 if i == proposed_index else 165
    p95_size = 195 if i == proposed_index else 155

    # Mean-to-P95 connection line
    ax_e.plot(
        [x_e[i], x_e[i]],
        [all_latency_mean[i], all_latency_p95[i]],
        color=colors[i],
        linewidth=line_width,
        zorder=2,
    )

    # Mean marker
    ax_e.scatter(
        x_e[i],
        all_latency_mean[i],
        s=mean_size,
        marker="o",
        color=colors[i],
        edgecolor="#444444",
        linewidth=1.8,
        zorder=4,
    )

    # P95 marker
    ax_e.scatter(
        x_e[i],
        all_latency_p95[i],
        s=p95_size,
        marker="D",
        color=colors[i],
        edgecolor="#444444",
        linewidth=1.8,
        zorder=4,
    )


# ------------------------------------------------------------
# Axis limits
# ------------------------------------------------------------
ax_e.set_ylim(2000, 5500)
ax_e.set_xlim(-0.30, len(methods) - 0.70)


# ------------------------------------------------------------
# Numeric annotations
# ------------------------------------------------------------
latency_span = 5500 - 2000

for i in range(len(methods)):
    # Mean value below circular marker
    ax_e.text(
        x_e[i],
        all_latency_mean[i] - 0.035 * latency_span,
        f"{all_latency_mean[i]:.0f}",
        ha="center",
        va="top",
        fontsize=VALUE_LABEL_SIZE,
    )

    # P95 value above diamond marker
    ax_e.text(
        x_e[i],
        all_latency_p95[i] + 0.025 * latency_span,
        f"{all_latency_p95[i]:.0f}",
        ha="center",
        va="bottom",
        fontsize=VALUE_LABEL_SIZE,
    )


# ------------------------------------------------------------
# PROP mean-latency reference line
# ------------------------------------------------------------
ax_e.hlines(
    y=all_latency_mean[proposed_index],
    xmin=ax_e.get_xlim()[0],
    xmax=x_e[proposed_index],
    colors=proposed_color,
    linestyles="--",
    linewidth=2.3,
    alpha=0.90,
    zorder=1,
)


# ------------------------------------------------------------
# Title and axes
# ------------------------------------------------------------
ax_e.set_title(
    "(e) Overall Closure Latency",
    fontsize=PANEL_TITLE_SIZE,
    pad=14,
)

ax_e.set_ylabel(
    "Ground Closure Latency (s)",
    fontsize=AXIS_LABEL_SIZE,
    labelpad=12,
)

ax_e.set_xticks(x_e)

ax_e.set_xticklabels(
    methods,
    rotation=30,
    ha="right",
    fontsize=METHOD_LABEL_SIZE,
)

ax_e.yaxis.set_major_locator(
    MultipleLocator(500)
)

ax_e.yaxis.set_minor_locator(
    MultipleLocator(250)
)

ax_e.tick_params(
    axis="both",
    which="major",
    labelsize=TICK_NUMBER_SIZE,
    width=1.8,
    length=7,
)


# ------------------------------------------------------------
# Legend: reduced by approximately 20%
# ------------------------------------------------------------
latency_legend = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor="#BBBBBB",
        markeredgecolor="#444444",

        # Original 1.4 × 0.8
        markeredgewidth=1.12,

        # Original 20 × 0.8
        markersize=16,

        label="Mean",
    ),
    Line2D(
        [0],
        [0],
        marker="D",
        linestyle="none",
        markerfacecolor="#BBBBBB",
        markeredgecolor="#444444",

        # Original 1.4 × 0.8
        markeredgewidth=1.12,

        # Original 20 × 0.8
        markersize=16,

        label="P95",
    ),
]


legend_e = ax_e.legend(
    handles=latency_legend,

    # Upper-right, but leave a visible margin from top/right borders
    loc="upper right",
    bbox_to_anchor=(0.975, 0.995),

    # Stack Mean and P95 vertically
    ncol=1,

    frameon=True,

    # Current legend font reduced by 20%
    fontsize=max(
        1,
        round(E_LEGEND_FONT_SIZE * 0.80),
    ),

    # Compact marker/text geometry
    handlelength=0.68,
    handletextpad=0.28,

    # Moderate vertical spacing between Mean and P95
    labelspacing=0.38,

    # Compact internal padding
    borderpad=0.32,

    # Keep a small distance from axes boundary
    borderaxespad=0.12,
)


# ------------------------------------------------------------
# Legend frame
# ------------------------------------------------------------
legend_e.get_frame().set_edgecolor(
    LEGEND_FRAME_EDGE_COLOR
)

legend_e.get_frame().set_linewidth(
    LEGEND_FRAME_LINEWIDTH
)

legend_e.get_frame().set_facecolor(
    LEGEND_FRAME_FACE_COLOR
)

legend_e.get_frame().set_alpha(
    LEGEND_FRAME_ALPHA
)

legend_e.set_zorder(20)
"""

# ============================================================
# 10. Panel (d): Utility-resource-downlink tradeoff
# ============================================================
bubble_scale = 2.6
bubble_sizes = downlink_mb * bubble_scale


# ------------------------------------------------------------
# Draw bubbles
# ------------------------------------------------------------
for i in range(len(methods)):
    edge_lw = 2.8 if i == proposed_index else 1.7

    ax_f.scatter(
        energy_wh[i],
        hv_net_utility[i],
        s=bubble_sizes[i],
        color=colors[i],
        edgecolor="#555555",
        linewidth=edge_lw,
        alpha=0.88,
        zorder=4,
    )


# ------------------------------------------------------------
# Method labels
#
# (-12, 12):
#   x = -12 points -> move left
#   y = +12 points -> move upward
#
# ha="right", va="bottom" ensures that the text body lies
# to the upper-left of the bubble center.
# ------------------------------------------------------------
method_label_offsets = {
    "Myopic": (-12, 12),
    "B1": (-12, 12),
    "OFFLOAD": (-12, 12),
    "TUT": (-12, 12),
    "PROP": (-12, 12),
}

for i, method in enumerate(methods):
    offset_x, offset_y = method_label_offsets[method]

    ax_f.annotate(
        method_labels_singleline[i],
        xy=(
            energy_wh[i],
            hv_net_utility[i],
        ),
        xytext=(
            offset_x,
            offset_y,
        ),
        textcoords="offset points",

        # Put labels at upper-left
        ha="right",
        va="bottom",

        fontsize=ANNOT_SIZE * 1.6,
        color="black",
        zorder=10,
        annotation_clip=False,
    )


# ------------------------------------------------------------
# Panel title and axis labels
# ------------------------------------------------------------
ax_f.set_title(
    "Utility–Resource Tradeoff",
    fontsize=PANEL_TITLE_SIZE,
    pad=14,
)

ax_f.set_xlabel(
    "Total Controllable Energy (Wh)",
    fontsize=X_AXIS_LABEL_SIZE,
    labelpad=11,
)

ax_f.set_ylabel(
    r"High-Value Net Utility $J_{\mathrm{HV}}^{\mathrm{net}}$",
    fontsize=AXIS_LABEL_SIZE,
    labelpad=12,
)


# ------------------------------------------------------------
# Automatic axis limits
# ------------------------------------------------------------
xmin_f, xmax_f, xstep_f = compute_axis_limits(
    energy_wh,
    include_zero=False,
    n_ticks=6,
    pad_ratio=0.10,
)

ymin_f, ymax_f, ystep_f = compute_axis_limits(
    hv_net_utility,
    include_zero=True,
    n_ticks=6,
    pad_ratio=0.14,
)

ax_f.set_xlim(
    xmin_f,
    xmax_f,
)

# Leave space for the upper-right bubble legend
ymax_f = max(
    ymax_f,
    60.0,
)

ax_f.set_ylim(
    ymin_f,
    ymax_f,
)

ax_f.xaxis.set_major_locator(
    MultipleLocator(xstep_f)
)

ax_f.xaxis.set_minor_locator(
    AutoMinorLocator(2)
)

ax_f.yaxis.set_major_locator(
    MultipleLocator(ystep_f)
)

ax_f.yaxis.set_minor_locator(
    AutoMinorLocator(2)
)

ax_f.tick_params(
    axis="both",
    which="major",
    labelsize=TICK_NUMBER_SIZE,
    width=1.8,
    length=7,
)


# ============================================================
# 14. Bubble-size legend
# ============================================================

# Reference downlink volumes shown in the legend
reference_volumes = [
    200,
    500,
    900,
]


# ------------------------------------------------------------
# Bubble legend parameters
# ------------------------------------------------------------

# Reduce the complete bubble legend by 10% without changing data bubbles.
LEGEND_SIZE_FACTOR = 0.90
LEGEND_BUBBLE_SCALE = 0.85 * LEGEND_SIZE_FACTOR ** 2

# Font size of "200 MB", "500 MB", and "900 MB"
LEGEND_NUMBER_SIZE = BUBBLE_LEGEND_SIZE * 1.6 * LEGEND_SIZE_FACTOR

# Font size of "Downlink volume"
LEGEND_TITLE_SIZE = BUBBLE_TITLE_SIZE * 1.5 * LEGEND_SIZE_FACTOR

# Vertical spacing between the three rows
LEGEND_VERTICAL_SPACING = 0.85

# Horizontal gap between bubble and number
LEGEND_CIRCLE_TEXT_GAP = 0.27

# Width reserved for bubble marker
LEGEND_HANDLE_LENGTH = 2.80 * LEGEND_SIZE_FACTOR

# Legend position
LEGEND_POSITION = (
    1.0,
    0.995,
)


# ------------------------------------------------------------
# Generate legend handles only once
# ------------------------------------------------------------
bubble_legend_handles = []

for volume in reference_volumes:
    scatter_area = (
        volume
        * bubble_scale
        * LEGEND_BUBBLE_SCALE
    )

    # scatter uses area, while Line2D markersize uses diameter
    marker_diameter = (
        2.0
        * np.sqrt(
            scatter_area / np.pi
        )
    )

    bubble_legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="#555555",
            markeredgewidth=1.4,
            markersize=marker_diameter,
            label=f"{volume} MB",
        )
    )


# ------------------------------------------------------------
# Draw bubble legend
# ------------------------------------------------------------
bubble_legend = ax_f.legend(
    handles=bubble_legend_handles,
    title="Downlink volume",

    loc="upper right",
    bbox_to_anchor=LEGEND_POSITION,

    frameon=True,

    fontsize=LEGEND_NUMBER_SIZE,
    title_fontsize=LEGEND_TITLE_SIZE,

    labelspacing=LEGEND_VERTICAL_SPACING,
    handletextpad=LEGEND_CIRCLE_TEXT_GAP,
    handlelength=LEGEND_HANDLE_LENGTH,

    borderpad=0.30,
    borderaxespad=0.08,

    alignment="left",
)


# ------------------------------------------------------------
# Bubble legend frame
# ------------------------------------------------------------
bubble_legend.get_frame().set_edgecolor(
    LEGEND_FRAME_EDGE_COLOR
)

bubble_legend.get_frame().set_linewidth(
    LEGEND_FRAME_LINEWIDTH
)

bubble_legend.get_frame().set_facecolor(
    LEGEND_FRAME_FACE_COLOR
)

bubble_legend.get_frame().set_alpha(
    LEGEND_FRAME_ALPHA
)

bubble_legend.set_zorder(20)
# ============================================================
# 15. Common journal-style formatting
# ============================================================
for ax in all_plot_axes:
    # 六个子图四周均显示黑色边框
    for spine_name in [
        "top",
        "right",
        "left",
        "bottom",
    ]:
        ax.spines[spine_name].set_visible(True)
        ax.spines[spine_name].set_color("black")
        ax.spines[spine_name].set_linewidth(2.0)

    # 不显示网格
    ax.grid(False)

    # 明显可见的内向刻度：保留向内，同时增强可见性
    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        color="#333333",
        labelcolor="black",
        width=1.05,
        length=5.0,
    )

    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        color="#555555",
        labelcolor="black",
        width=0.80,
        length=3.0,
    )


# Put each panel index directly below its own subplot.
for ax, panel_index in zip(
    all_plot_axes,
    ["(a)", "(b)", "(c)", "(d)"],
):
    ax.text(
        0.5,
        -0.30,
        panel_index,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=PANEL_INDEX_SIZE,
        fontweight="bold",
        clip_on=False,
    )


# ============================================================
# 16. Final layout
# ============================================================
fig.subplots_adjust(
    left=0.085,
    right=0.985,
    top=0.965,
    bottom=0.125,
)


# ============================================================
# 17. Export
# ============================================================
pdf_path = (
    OUTPUT_DIR
    / "main_four_panel.pdf"
)

png_path = (
    OUTPUT_DIR
    / "main_four_panel.png"
)


# Vector PDF
fig.savefig(
    pdf_path,
    bbox_inches="tight",
    facecolor="white",
    metadata={
        "Creator": "leo_onboard_codesign_case_study_v2",
        "CreationDate": None,
        "ModDate": None,
    },
)


# High-resolution PNG
fig.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)


print(
    f"Vector PDF saved to: "
    f"{pdf_path.resolve()}"
)

print(
    f"High-resolution PNG saved to: "
    f"{png_path.resolve()}"
)


# 使用无界面后端时关闭图形，释放内存
plt.close(fig)
