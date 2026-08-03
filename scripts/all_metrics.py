from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


# ============================================================
# 1. Output settings
# ============================================================
OUTPUT_DIR = Path(
    "combined_primary_auxiliary_metrics_final"
)
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. Methods and fixed colors
# ============================================================
methods = [
    "Myopic",
    "B1",
    "OFFLOAD",
    "TUT",
    "PROP",
]

method_labels_singleline = [
    "Myopic",
    "B1",
    "OFFLOAD",
    "TUT",
    "PROP",
]

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
# 3. Primary-metric data
# ============================================================
primary_metrics = [
    {
        "panel": "(a)",
        "title": "Overall Net Utility",
        "ylabel": (
            r"Overall Task Net Utility "
            r"$J_{\mathrm{all}}^{\mathrm{net}}$"
        ),
        "means": np.array([
            38.923,
            80.898,
            77.632,
            88.840,
            85.401,
        ]),
        "stds": np.array([
            1.120,
            3.606,
            2.820,
            2.201,
            2.294,
        ]),
        "ylim": (0, 108),
        "major_tick": 20,
        "minor_tick": 10,
        "value_decimals": 2,
    },
    {
        "panel": "(b)",
        "title": "High-Value Net Utility",
        "ylabel": (
            r"High-Value Net Utility "
            r"$J_{\mathrm{HV}}^{\mathrm{net}}$"
        ),
        "means": np.array([
            8.645,
            57.984,
            46.393,
            48.596,
            60.444,
        ]),
        "stds": np.array([
            1.763,
            3.148,
            2.986,
            1.246,
            3.817,
        ]),
        "ylim": (0, 72),
        "major_tick": 15,
        "minor_tick": 5,
        "value_decimals": 2,
    },
    {
        "panel": "(c)",
        "title": "High-Value Task Closures",
        "ylabel": (
            r"High-Value Task Closures "
            r"$N_{\mathrm{HV}}^{\mathrm{closure}}$"
        ),
        "means": np.array([
            8.6,
            44.8,
            44.0,
            28.6,
            44.6,
        ]),
        "stds": np.array([
            1.1,
            2.7,
            2.3,
            1.9,
            2.1,
        ]),
        "ylim": (0, 55),
        "major_tick": 10,
        "minor_tick": 5,
        "value_decimals": 1,
    },
]


# ============================================================
# 4. Auxiliary-metric data
# ============================================================

# Panel (d): non-exclusive HV layer-delivery counts
hv_metadata = np.array([
    37.0,
    0.0,
    0.0,
    4.0,
    45.0,
])

hv_roi = np.array([
    4.6,
    0.0,
    3.2,
    28.8,
    6.6,
])

hv_evidence = np.array([
    4.0,
    44.8,
    40.8,
    22.0,
    37.8,
])

hv_total = (
    hv_metadata
    + hv_roi
    + hv_evidence
)


# Panel (e): overall closure latency, unit: seconds
all_latency_mean = np.array([
    2580,
    3690,
    2947,
    2472,
    2501,
])

all_latency_p95 = np.array([
    4400,
    8421,
    5012,
    4420,
    4412,
])


# Panel (f): utility-resource-downlink tradeoff
hv_net_utility = np.array([
    8.645,
    57.984,
    46.393,
    48.596,
    60.444,
])

energy_wh = np.array([
    1.825,
    4.177,
    6.074,
    2.421,
    2.210,
])

downlink_mb = np.array([
    119.857,
    676.000,
    487.650,
    142.111,
    184.694,
])


# ============================================================
# 5. Publication font settings
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "stix",
    "axes.linewidth": 2.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "hatch.linewidth": 1.4,
})

available_fonts = {
    font.name
    for font in font_manager.fontManager.ttflist
}

if "Times New Roman" not in available_fonts:
    print(
        "Warning: Times New Roman was not found. "
        "Matplotlib will use the configured serif fallback."
    )


# ============================================================
# 6. Font sizes
#    Keep every enlarged value from the user's current code
# ============================================================
FONT_SCALE = 1.30

ROW_TITLE_SIZE = round(
    28 * FONT_SCALE
)

PANEL_TITLE_SIZE = round(
    22 * FONT_SCALE
)

AXIS_LABEL_SIZE = round(
    19 * FONT_SCALE
)

X_AXIS_LABEL_SIZE = round(
    21 * FONT_SCALE
)

# Method names: retain the additional 35% enlargement
METHOD_LABEL_SIZE = round(
    17 * FONT_SCALE * 1.35
)

# Tick numbers: retain the additional 50% enlargement
TICK_NUMBER_SIZE = round(
    17 * FONT_SCALE * 1.50
)

# Data labels: retain the additional 50% enlargement
VALUE_LABEL_SIZE = round(
    15 * FONT_SCALE * 1.50
)

LEGEND_SIZE = round(
    14 * FONT_SCALE
)

NOTE_SIZE = round(
    13 * FONT_SCALE
)

ANNOT_SIZE = round(
    14 * FONT_SCALE
)

BUBBLE_LEGEND_SIZE = round(
    13 * FONT_SCALE
)

BUBBLE_TITLE_SIZE = round(
    14 * FONT_SCALE
)

# Panel (d): twice the current legend font
LAYER_LEGEND_SIZE = round(
    LEGEND_SIZE * 2.0
)

# Panel (e): twice the current legend font
LATENCY_LEGEND_SIZE = round(
    LEGEND_SIZE * 2.0
)


# ============================================================
# 7. Create four-row GridSpec
#
# Row 0: Primary Metrics heading
# Row 1: Primary plots
# Row 2: Auxiliary Metrics heading
# Row 3: Auxiliary plots
# ============================================================
fig = plt.figure(
    figsize=(28, 19.2),
    facecolor="white",
)

grid = fig.add_gridspec(
    nrows=4,
    ncols=3,
    height_ratios=[
        0.085,
        1.00,
        0.085,
        1.00,
    ],
    width_ratios=[
        1.05,
        1.05,
        1.05,
    ],
    hspace=0.34,
    wspace=0.34,
)


# ============================================================
# 8. Row headings
# ============================================================
primary_heading_ax = fig.add_subplot(
    grid[0, :]
)
primary_heading_ax.axis("off")

primary_heading_ax.text(
    0.5,
    0.50,
    "Primary Metrics",
    ha="center",
    va="center",
    fontsize=ROW_TITLE_SIZE,
    fontweight="bold",
)


auxiliary_heading_ax = fig.add_subplot(
    grid[2, :]
)
auxiliary_heading_ax.axis("off")

# Move Auxiliary Metrics downward
auxiliary_heading_ax.text(
    0.5,
    0.06,
    "Auxiliary Metrics",
    ha="center",
    va="center",
    fontsize=ROW_TITLE_SIZE,
    fontweight="bold",
)


# ============================================================
# 9. Create six plotting axes
# ============================================================
primary_axes = [
    fig.add_subplot(
        grid[1, column]
    )
    for column in range(3)
]

ax_d = fig.add_subplot(
    grid[3, 0]
)

ax_e = fig.add_subplot(
    grid[3, 1]
)

ax_f = fig.add_subplot(
    grid[3, 2]
)

all_plot_axes = [
    *primary_axes,
    ax_d,
    ax_e,
    ax_f,
]


# ============================================================
# 10. Top row: primary metrics
# ============================================================
for ax, metric in zip(
    primary_axes,
    primary_metrics,
):
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

    # Fixed horizontal range
    # The left boundary is exactly the y-axis position
    x_left = -0.55
    x_right = len(methods) - 0.45

    ax.set_xlim(
        x_left,
        x_right,
    )

    # PROP reference line:
    # xmin=0 means the left y-axis;
    # xmax is converted into an axes-fraction coordinate.
    prop_x_fraction = (
        (
            x[proposed_index]
            - x_left
        )
        /
        (
            x_right
            - x_left
        )
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

    y_min, y_max = metric["ylim"]

    vertical_offset = (
        0.024
        * (y_max - y_min)
    )

    for bar, mean, std in zip(
        bars,
        means,
        stds,
    ):
        x_center = (
            bar.get_x()
            + bar.get_width() / 2
        )

        y_position = (
            mean
            + std
            + vertical_offset
        )

        ax.text(
            x_center,
            y_position,
            f"{mean:.{metric['value_decimals']}f}",
            ha="center",
            va="bottom",
            fontsize=VALUE_LABEL_SIZE,
            color="black",
        )

    ax.set_title(
        (
            f"{metric['panel']} "
            f"{metric['title']}"
        ),
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

    ax.set_ylim(
        *metric["ylim"]
    )

    ax.yaxis.set_major_locator(
        MultipleLocator(
            metric["major_tick"]
        )
    )

    ax.yaxis.set_minor_locator(
        MultipleLocator(
            metric["minor_tick"]
        )
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
#     Keep each method's original color
# ============================================================
bar_width = 0.64

for i in range(len(methods)):
    edge_lw = (
        2.6
        if i == proposed_index
        else 1.5
    )

    # Metadata
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

    # ROI
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

    # Evidence
    ax_d.bar(
        x[i],
        hv_evidence[i],
        width=bar_width,
        bottom=(
            hv_metadata[i]
            + hv_roi[i]
        ),
        color=colors[i],
        edgecolor="#666666",
        linewidth=edge_lw,
        alpha=0.95,
        zorder=3,
    )


# Segment and total values
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
            (
                hv_metadata[i]
                + hv_roi[i] / 2
            ),
            f"{hv_roi[i]:.1f}",
            ha="center",
            va="center",
            fontsize=VALUE_LABEL_SIZE,
        )

    if hv_evidence[i] >= 3:
        ax_d.text(
            x[i],
            (
                hv_metadata[i]
                + hv_roi[i]
                + hv_evidence[i] / 2
            ),
            f"{hv_evidence[i]:.1f}",
            ha="center",
            va="center",
            fontsize=VALUE_LABEL_SIZE,
        )

    ax_d.text(
        x[i],
        hv_total[i] + 2.1,
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

ax_d.set_xlim(
    -0.55,
    len(methods) - 0.45,
)

ax_d.set_ylim(
    0,
    100,
)

ax_d.yaxis.set_major_locator(
    MultipleLocator(20)
)

ax_d.yaxis.set_minor_locator(
    MultipleLocator(10)
)

ax_d.tick_params(
    axis="both",
    which="major",
    labelsize=TICK_NUMBER_SIZE,
    width=1.8,
    length=7,
)


# Neutral legend swatches explain layer hatches;
# actual bars retain method-specific colors.
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

legend_d = ax_d.legend(
    handles=layer_legend,
    loc="upper left",
    bbox_to_anchor=(
        0.005,
        0.995,
    ),
    frameon=False,
    fontsize=LAYER_LEGEND_SIZE,
    handlelength=1.30,
    handletextpad=0.35,
    labelspacing=0.25,
    borderaxespad=0.10,
)

legend_d.set_zorder(20)


# ============================================================
# 12. Panel (e): Overall closure latency
# ============================================================
x_e = np.arange(
    len(methods)
)

for i in range(len(methods)):
    line_width = (
        3.8
        if i == proposed_index
        else 2.7
    )

    mean_size = (
        205
        if i == proposed_index
        else 165
    )

    p95_size = (
        195
        if i == proposed_index
        else 155
    )

    ax_e.plot(
        [
            x_e[i],
            x_e[i],
        ],
        [
            all_latency_mean[i],
            all_latency_p95[i],
        ],
        color=colors[i],
        linewidth=line_width,
        zorder=2,
    )

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

    ax_e.text(
        x_e[i],
        all_latency_mean[i] - 120,
        f"{all_latency_mean[i]:.0f}",
        ha="center",
        va="top",
        fontsize=VALUE_LABEL_SIZE,
    )

    ax_e.text(
        x_e[i],
        all_latency_p95[i] + 120,
        f"{all_latency_p95[i]:.0f}",
        ha="center",
        va="bottom",
        fontsize=VALUE_LABEL_SIZE,
    )


# PROP mean-latency reference line
ax_e.set_xlim(
    -0.30,
    len(methods) - 0.70,
)

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

ax_e.set_ylim(
    1500,
    9000,
)

ax_e.yaxis.set_major_locator(
    MultipleLocator(1000)
)

ax_e.yaxis.set_minor_locator(
    MultipleLocator(500)
)

ax_e.tick_params(
    axis="both",
    which="major",
    labelsize=TICK_NUMBER_SIZE,
    width=1.8,
    length=7,
)


latency_legend = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor="#BBBBBB",
        markeredgecolor="#444444",
        markeredgewidth=1.4,
        markersize=20,
        label="Mean",
    ),
    Line2D(
        [0],
        [0],
        marker="D",
        linestyle="none",
        markerfacecolor="#BBBBBB",
        markeredgecolor="#444444",
        markeredgewidth=1.4,
        markersize=20,
        label="P95",
    ),
]

legend_e = ax_e.legend(
    handles=latency_legend,
    loc="upper right",
    bbox_to_anchor=(
        0.985,
        0.985,
    ),
    ncol=2,
    frameon=False,
    fontsize=LATENCY_LEGEND_SIZE,
    handlelength=0.55,
    handletextpad=0.10,
    columnspacing=0.55,
    borderaxespad=0.0,
    labelspacing=0.15,
)

legend_e.set_zorder(20)


# ============================================================
# 13. Panel (f): Utility-resource-downlink tradeoff
# ============================================================
bubble_scale = 2.6

bubble_sizes = (
    downlink_mb
    * bubble_scale
)

for i in range(len(methods)):
    edge_lw = (
        2.8
        if i == proposed_index
        else 1.7
    )

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


# Method labels at lower-right of each bubble
for i in range(len(methods)):
    bubble_radius = np.sqrt(
        bubble_sizes[i] / np.pi
    )

    offset_x = (
        0.72
        * bubble_radius
        + 4
    )

    offset_y = -(
        0.72
        * bubble_radius
        + 4
    )

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
        ha="left",
        va="top",
        fontsize=ANNOT_SIZE * 1.6,
        color="black",
        zorder=10,
        annotation_clip=False,
    )


ax_f.set_title(
    "(f) Utility–Resource Tradeoff",
    fontsize=PANEL_TITLE_SIZE,
    pad=14,
)

ax_f.set_xlabel(
    "Total Controllable Energy (Wh)",
    fontsize=X_AXIS_LABEL_SIZE,
    labelpad=11,
)

ax_f.set_ylabel(
    (
        r"High-Value Net Utility "
        r"$J_{\mathrm{HV}}^{\mathrm{net}}$"
    ),
    fontsize=AXIS_LABEL_SIZE,
    labelpad=12,
)

ax_f.set_xlim(
    1.1,
    7.75,
)

ax_f.set_ylim(
    0,
    70,
)

ax_f.xaxis.set_major_locator(
    MultipleLocator(1)
)

ax_f.xaxis.set_minor_locator(
    MultipleLocator(0.5)
)

ax_f.yaxis.set_major_locator(
    MultipleLocator(10)
)

ax_f.yaxis.set_minor_locator(
    MultipleLocator(5)
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
reference_volumes = [
    150,
    400,
    650,
]

bubble_legend_handles = []

for volume in reference_volumes:
    scatter_area = (
        volume
        * bubble_scale
    )

    marker_diameter = (
        2.0
        * np.sqrt(
            scatter_area
            / np.pi
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


bubble_legend = ax_f.legend(
    handles=bubble_legend_handles,
    title="Downlink volume",
    loc="lower right",
    bbox_to_anchor=(
        0.985,
        0.068,
    ),
    frameon=False,
    fontsize=BUBBLE_LEGEND_SIZE,
    title_fontsize=BUBBLE_TITLE_SIZE,
    labelspacing=1.75,
    handletextpad=0.72,
    handlelength=2.30,
    borderpad=0.05,
    borderaxespad=0.0,
    alignment="left",
)

bubble_legend.set_zorder(20)


# ============================================================
# 15. Common journal-style formatting
# ============================================================
for ax in all_plot_axes:
    ax.spines["top"].set_visible(
        False
    )

    ax.spines["right"].set_visible(
        False
    )

    ax.spines["left"].set_visible(
        True
    )

    ax.spines["bottom"].set_visible(
        True
    )

    ax.spines["left"].set_linewidth(
        2.0
    )

    ax.spines["bottom"].set_linewidth(
        2.0
    )

    ax.grid(
        False
    )





# ============================================================
# 17. Final layout
# ============================================================
fig.subplots_adjust(
    left=0.055,
    right=0.992,
    top=0.985,
    bottom=0.095,
)


# ============================================================
# 18. Export
# ============================================================
pdf_path = (
    OUTPUT_DIR
    / "primary_and_auxiliary_metrics_final.pdf"
)

png_path = (
    OUTPUT_DIR
    / "primary_and_auxiliary_metrics_final.png"
)

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

print(
    f"Vector PDF saved to: "
    f"{pdf_path.resolve()}"
)

print(
    f"High-resolution PNG saved to: "
    f"{png_path.resolve()}"
)

plt.show()