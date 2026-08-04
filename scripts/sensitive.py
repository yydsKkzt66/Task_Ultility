from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Rectangle, ConnectionPatch

# ============================================================
# 1. Output settings
# ============================================================
OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "sensitivity_ablation_2x2_zoom_legends_v1"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SHOW_FIGURE = False


# ============================================================
# 2. Color palette
# ============================================================
PALETTE = {
    "myopic_blue": "#B2E6FD",
    "b1_green": "#B8D2CC",
    "offload_orange": "#E8B2A7",
    "tut_yellow": "#FEEBB9",
    "prop_purple": "#B5AED5",
}

# Shared metric colors (for utility/HV-closure related metrics)
METRIC_STYLES = {
    "J_all": {
        "label": r"$J_{\mathrm{all}}^{\mathrm{net}}$",
        "color": PALETTE["myopic_blue"],
        "marker": "o",
    },
    "J_HV": {
        "label": r"$J_{\mathrm{HV}}^{\mathrm{net}}$",
        "color": PALETTE["prop_purple"],
        "marker": "s",
    },
    "N_HV": {
        "label": r"$N_{\mathrm{HV}}^{\mathrm{closure}}$",
        "color": PALETTE["b1_green"],
        "marker": "^",
    },
}

# Resource colors for bottom-right panel: use the new 3-color SCI palette
RESOURCE_STYLES = {
    "energy": {
        "label": "Energy",
        "color": "#F08F92",
        "hatch": "",
    },
    "downlink": {
        "label": "Downlink",
        "color": "#9CBEDB",
        "hatch": "//",
    },
    "queue": {
        "label": "Peak queue",
        "color": "#A9D5A5",
        "hatch": "xx",
    },
}


def darken(hex_color: str, factor: float = 0.68):
    rgb = np.array(mcolors.to_rgb(hex_color))
    return tuple(np.clip(rgb * factor, 0.0, 1.0))


# ============================================================
# 3. Sensitivity data (5 seeds, mean ± std)
#    Reference scenario: V25, cap = 0.50
# ============================================================
# Capacity sensitivity under V25
capacity_x = np.array([0.25, 0.35, 0.50, 0.70])

capacity_mean = {
    "J_all": np.array([42.103, 43.682, 44.909, 45.732]),
    "J_HV": np.array([35.641, 36.875, 37.921, 38.576]),
    "N_HV": np.array([56.0, 56.0, 56.2, 56.2]),
}

capacity_std = {
    "J_all": np.array([5.281, 5.376, 5.465, 5.608]),
    "J_HV": np.array([4.953, 5.007, 5.099, 5.158]),
    "N_HV": np.array([2.74, 2.74, 3.11, 3.11]),
}

# Value heterogeneity sensitivity under cap = 0.50
value_labels = ["V10", "V25", "V50"]
value_x = np.arange(len(value_labels))

value_mean = {
    "J_all": np.array([40.413, 44.909, 47.255]),
    "J_HV": np.array([30.464, 37.921, 41.604]),
    "N_HV": np.array([56.2, 56.2, 56.6]),
}

value_std = {
    "J_all": np.array([4.544, 5.465, 5.943]),
    "J_HV": np.array([3.773, 5.099, 5.662]),
    "N_HV": np.array([3.11, 3.11, 3.05]),
}


# ============================================================
# 4. Ablation data (5 seeds, mean ± std)
# ============================================================
ablation_labels = [
    "Full",
    "w/o Lyapunov",
    "w/o Mode",
    "Fixed-B",
    "w/o Prog.",
]
ablation_x = np.arange(len(ablation_labels))

ablation_mean = {
    "J_all": np.array([44.909, 29.178, 44.524, 40.699, 38.408]),
    "J_HV": np.array([37.921, 19.227, 37.539, 34.044, 33.726]),
    "N_HV": np.array([56.2, 59.2, 41.0, 55.6, 30.2]),
}

ablation_std = {
    "J_all": np.array([5.465, 5.328, 4.878, 4.782, 5.871]),
    "J_HV": np.array([5.099, 4.239, 4.661, 4.379, 5.771]),
    "N_HV": np.array([3.1, 3.0, 3.7, 3.2, 4.4]),
}

resource_mean = {
    "energy": np.array([2.339, 6.023, 1.431, 1.641, 1.382]),
    "downlink": np.array([167.190, 451.794, 110.879, 166.071, 62.444]),
    "queue": np.array([8.853, 50.785, 8.074, 8.725, 8.846]),
}

resource_std = {
    "energy": np.zeros(5),
    "downlink": np.zeros(5),
    "queue": np.zeros(5),
}


# ============================================================
# 5. Normalize to the main setting: V25, cap=0.50 = 100%
# ============================================================
REFERENCE_MEAN = {
    "J_all": 44.909,
    "J_HV": 37.921,
    "N_HV": 56.2,
    "energy": 2.339,
    "downlink": 167.190,
    "queue": 8.853,
}


def normalize(means, stds, keys):
    normalized_mean = {}
    normalized_std = {}
    for key in keys:
        ref = REFERENCE_MEAN[key]
        normalized_mean[key] = means[key] / ref * 100.0
        normalized_std[key] = stds[key] / ref * 100.0
    return normalized_mean, normalized_std


capacity_norm, capacity_err = normalize(
    capacity_mean, capacity_std, ["J_all", "J_HV", "N_HV"]
)
value_norm, value_err = normalize(
    value_mean, value_std, ["J_all", "J_HV", "N_HV"]
)
ablation_norm, ablation_err = normalize(
    ablation_mean, ablation_std, ["J_all", "J_HV", "N_HV"]
)
resource_norm, resource_err = normalize(
    resource_mean, resource_std, ["energy", "downlink", "queue"]
)


# ============================================================
# 6. Font and style settings
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.15,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

AXIS_LABEL_SIZE = 10.8
TICK_LABEL_SIZE = 10.2
SMALL_TITLE_SIZE = 10.8
PANEL_LABEL_SIZE = 12.2
METRIC_LEGEND_SIZE = 10.8 * 0.90  # 10% smaller than the previous shared legend
METRIC_LEGEND_FRAME_SCALE = 0.85  # compact the legend box by 15%
RESOURCE_LEGEND_SIZE = 10.2
RESOURCE_LEGEND_TITLE_SIZE = 10.4

VALUE_LABEL_SIZE = 8.8


# ============================================================
# 7. Create 2×2 figure
# ============================================================
fig, axes = plt.subplots(
    2,
    2,
    figsize=(7.35, 6.45),
    facecolor="white",
)

ax_tl = axes[0, 0]
ax_tr = axes[0, 1]
ax_bl = axes[1, 0]
ax_br = axes[1, 1]

fig.subplots_adjust(
    left=0.085,
    right=0.985,
    top=0.955,
    bottom=0.115,
    wspace=0.24,
    hspace=0.42,
)


# ============================================================
# 8. Helper functions
# ============================================================
def style_axis(ax):
    ax.set_axisbelow(True)

    ax.grid(
        axis="both",
        which="major",
        color="#D8D8D8",
        linestyle="-",
        linewidth=0.60,
        alpha=0.85,
        zorder=0,
    )

    ax.grid(
        axis="both",
        which="minor",
        color="#ECECEC",
        linestyle="-",
        linewidth=0.40,
        alpha=0.75,
        zorder=0,
    )

    # fully boxed
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#4D4D4D")
        spine.set_linewidth(1.0)

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        labelsize=TICK_LABEL_SIZE,
        width=0.75,
        length=2.6,
        color="#666666",
        labelcolor="black",
    )

    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        width=0.55,
        length=1.5,
        color="#888888",
    )


def draw_sensitivity_panel(
    ax,
    x_values,
    means,
    errors,
    x_label,
    x_ticks,
    x_ticklabels,
    main_x,
    title_text,
    highlight_main=False,
):
    ax.axhline(
        100,
        color="#888888",
        linestyle=(0, (3, 2)),
        linewidth=0.95,
        zorder=1,
    )

    ax.axvline(
        main_x,
        color=PALETTE["prop_purple"],
        linestyle=(0, (2, 2)),
        linewidth=0.95,
        alpha=0.80,
        zorder=1,
    )

    for key, style in METRIC_STYLES.items():
        face = style["color"]
        edge = darken(face)

        ax.errorbar(
            x_values,
            means[key],
            yerr=errors[key],
            color=edge,
            marker=style["marker"],
            markersize=10 if key == "J_all" else 6.2,
            markerfacecolor=face,
            markeredgecolor=edge,   # no white border for J_all
            markeredgewidth=0.9,
            linewidth=1.55,
            elinewidth=0.9,
            capsize=2.6,
            capthick=0.9,
            zorder=3,
        )

    if highlight_main:
        # only highlight the main reference points, without adding the "Main" text
        for key, style in METRIC_STYLES.items():
            y = 100.0
            face = style["color"]
            edge = darken(face)
            ax.scatter(
                main_x,
                y,
                s=58,
                facecolors=face,
                edgecolors=edge,
                linewidths=0.9,
                zorder=7,
                marker=style["marker"],
            )

    ax.set_xlabel(x_label, fontsize=AXIS_LABEL_SIZE, labelpad=4)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels, fontsize=TICK_LABEL_SIZE)

    ax.set_ylim(68, 125)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))

    ax.text(
        0.02,
        0.96,
        title_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=SMALL_TITLE_SIZE,
    )

    style_axis(ax)



def add_capacity_zoom_inset(ax):
    """Magnify the capacity interval 0.29--0.31 in the top-left panel.

    The original sensitivity samples are at 0.25 and 0.35 around this
    interval, so the inset displays the linearly interpolated mean trends
    between those two measured points. No additional experimental sample
    is introduced.
    """
    zoom_xmin, zoom_xmax = 0.2993, 0.3007
    zoom_x = np.linspace(zoom_xmin, zoom_xmax, 81)

    # Use the blank upper-right area created by the 125% y-axis limit.
    axins = ax.inset_axes([0.25, 0.10, 0.34, 0.23])
    axins.set_facecolor("white")
    axins.set_zorder(12)

    zoom_values = []
    for key in ["J_all", "J_HV"]:
        style = METRIC_STYLES[key]
        face = style["color"]
        edge = darken(face)
        y_zoom = np.interp(zoom_x, capacity_x, capacity_norm[key])
        y_mid = float(np.interp(0.30, capacity_x, capacity_norm[key]))
        zoom_values.extend(y_zoom.tolist())

        axins.plot(
            zoom_x,
            y_zoom,
            color=edge,
            linewidth=1.35,
            zorder=3,
        )
        axins.scatter(
            [0.30],
            [y_mid],
            s=22,
            marker=style["marker"],
            facecolor=face,
            edgecolor=edge,
            linewidth=0.65,
            zorder=4,
        )

    axins.set_xlim(0.2993, 0.3007)
    axins.set_ylim(95.45, 95.68)

    axins.set_xticks([0.2993, 0.3000, 0.3007])
    axins.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    axins.xaxis.set_minor_locator(AutoMinorLocator(2))

    axins.yaxis.set_major_locator(MultipleLocator(0.05))
    axins.yaxis.set_minor_locator(MultipleLocator(0.01))
    axins.tick_params(
        axis="both",
        which="major",
        direction="in",
        top=True,
        right=True,
        labelsize=7.0,
        width=0.55,
        length=1.8,
        color="#666666",
        labelcolor="black",
    )

    axins.tick_params(
        axis="both",
        which="minor",
        direction="in",
        top=True,
        right=True,
        width=0.40,
        length=1.0,
        color="#888888",
    )

    axins.grid(
        axis="both",
        which="major",
        color="#D8D8D8",
        linestyle="-",
        linewidth=0.45,
        alpha=0.85,
        zorder=0,
    )
    axins.grid(
        axis="both",
        which="minor",
        color="#ECECEC",
        linestyle="-",
        linewidth=0.30,
        alpha=0.75,
        zorder=0,
    )
    for spine in axins.spines.values():
        spine.set_color("#4D4D4D")
        spine.set_linewidth(0.85)

    axins.text(
        0.04,
        0.95,
        "Local detail",
        transform=axins.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
    )

    # Draw a visible source frame around the local region in the main panel.
    # The frame is intentionally wider than the numerical inset limits so it
    # remains legible at the scale of the full capacity axis.
    source_x0, source_x1 = 0.283, 0.317
    source_y0, source_y1 = 94.35, 96.80
    source_box = Rectangle(
        (source_x0, source_y0),
        source_x1 - source_x0,
        source_y1 - source_y0,
        facecolor="none",
        edgecolor="#666666",
        linewidth=0.90,
        linestyle=(0, (3, 2)),
        zorder=9,
    )
    ax.add_patch(source_box)

    # Project the right edge of the source frame to the left edge of the inset.
    # This produces a clean magnifying-window effect without crossing lines.
    connector_specs = [
        ((source_x1, source_y1), (0.0, 1.0)),
        ((source_x1, source_y0), (0.0, 0.0)),
    ]
    for source_xy, inset_xy in connector_specs:
        connector = ConnectionPatch(
            xyA=source_xy,
            coordsA=ax.transData,
            xyB=inset_xy,
            coordsB=axins.transAxes,
            arrowstyle="-",
            color="#666666",
            linewidth=0.80,
            linestyle=(0, (3, 2)),
            zorder=8,
            clip_on=False,
        )
        ax.figure.add_artist(connector)


def add_metric_legend(ax, loc, bbox_to_anchor, scale=1.0):
    """Add one compact metric legend inside a panel."""
    legend = ax.legend(
        metric_handles,
        metric_labels,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=1,
        frameon=True,
        fontsize=METRIC_LEGEND_SIZE * scale,
        handlelength=1.8 * METRIC_LEGEND_FRAME_SCALE * scale,
        handletextpad=0.5 * METRIC_LEGEND_FRAME_SCALE * scale,
        columnspacing=0.8 * METRIC_LEGEND_FRAME_SCALE * scale,
        labelspacing=0.45 * METRIC_LEGEND_FRAME_SCALE * scale,
        borderpad=0.42 * METRIC_LEGEND_FRAME_SCALE * scale,
        borderaxespad=0.20,
    )
    legend.get_frame().set_edgecolor("#4D4D4D")
    legend.get_frame().set_linewidth(0.9)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1.0)
    legend.set_zorder(20)
    return legend


def draw_grouped_bars(
    ax,
    labels,
    means,
    errors,
    style_dict,
    keys,
    title_text,
    ylabel,
    ylim,
    add_value_labels=False,
):
    n_groups = len(labels)
    n_series = len(keys)
    x = np.arange(n_groups)

    total_width = 0.78
    bar_width = total_width / n_series
    start = -total_width / 2 + bar_width / 2

    for idx, key in enumerate(keys):
        xx = x + start + idx * bar_width
        face = style_dict[key]["color"]
        edge = darken(face, 0.62)

        ax.bar(
            xx,
            means[key],
            width=bar_width,
            color=face,
            edgecolor=edge,
            linewidth=0.85,
            hatch=style_dict[key].get("hatch", ""),
            zorder=3,
        )

        if errors[key] is not None and np.any(errors[key] > 0):
            ax.errorbar(
                xx,
                means[key],
                yerr=errors[key],
                fmt="none",
                ecolor=edge,
                elinewidth=0.85,
                capsize=2.4,
                capthick=0.85,
                zorder=4,
            )

        if add_value_labels:
            for xi, yi in zip(xx, means[key]):
                ax.text(
                    xi,
                    yi + 1.3,
                    f"{yi:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=VALUE_LABEL_SIZE,
                )

    ax.axhline(
        100,
        color="#888888",
        linestyle=(0, (3, 2)),
        linewidth=0.95,
        zorder=1,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=24, ha="right", fontsize=TICK_LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE, labelpad=4)
    ax.set_ylim(*ylim)

    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(MultipleLocator(10))

    ax.text(
        0.02,
        0.96,
        title_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=SMALL_TITLE_SIZE,
    )

    style_axis(ax)


# ============================================================
# 9. Top-left: capacity sensitivity
# ============================================================
draw_sensitivity_panel(
    ax_tl,
    capacity_x,
    capacity_norm,
    capacity_err,
    "Downlink capacity multiplier",
    capacity_x,
    [f"{x:.2f}" for x in capacity_x],
    0.50,
    "Capacity variation",
    highlight_main=False,
)

ax_tl.set_ylabel("Relative performance (%)", fontsize=AXIS_LABEL_SIZE, labelpad=4)
add_capacity_zoom_inset(ax_tl)


# ============================================================
# 10. Top-right: value heterogeneity
# ============================================================
draw_sensitivity_panel(
    ax_tr,
    value_x,
    value_norm,
    value_err,
    "Value profile",
    value_x,
    value_labels,
    1,   # V25 is index 1
    "Value heterogeneity",
    highlight_main=True,
)


# ============================================================
# 11. Bottom-left: ablation performance
# ============================================================
draw_grouped_bars(
    ax_bl,
    ablation_labels,
    ablation_norm,
    ablation_err,
    METRIC_STYLES,
    ["J_all", "J_HV", "N_HV"],
    "Ablation performance",
    "Relative performance (%)",
    (40, 135),
    add_value_labels=False,
)


# ============================================================
# 12. Bottom-right: ablation resource burden
# ============================================================
draw_grouped_bars(
    ax_br,
    ablation_labels,
    resource_norm,
    resource_err,
    RESOURCE_STYLES,
    ["energy", "downlink", "queue"],
    "Resource burden",
    "Relative burden (%)",
    (0, 650),
    add_value_labels=False,
)

ax_br.yaxis.set_major_locator(MultipleLocator(100))
ax_br.yaxis.set_minor_locator(MultipleLocator(50))


# ============================================================
# 13. Separate metric legends inside top-left, top-right, and bottom-left
# ============================================================
metric_handles = []
metric_labels = []

for key in ["J_all", "J_HV", "N_HV"]:
    face = METRIC_STYLES[key]["color"]
    edge = darken(face)
    handle = plt.Line2D(
        [0],
        [0],
        color=edge,
        marker=METRIC_STYLES[key]["marker"],
        markerfacecolor=face,
        markeredgecolor=edge,
        markeredgewidth=0.8,
        linewidth=1.5,
        markersize=6.2,
        label=METRIC_STYLES[key]["label"],
    )
    metric_handles.append(handle)
    metric_labels.append(METRIC_STYLES[key]["label"])

# Top-left: keep the legend in the lower blank region so that it does
# not overlap the magnified inset in the upper-right region.
metric_legend_tl = add_metric_legend(
    ax_tl,
    loc="lower right",
    bbox_to_anchor=(0.985, 0.028),
)

# Top-right and bottom-left: use the upper-right blank region.
metric_legend_tr = add_metric_legend(
    ax_tr,
    loc="upper right",
    bbox_to_anchor=(0.985, 0.35),
)

metric_legend_bl = add_metric_legend(
    ax_bl,
    loc="upper right",
    bbox_to_anchor=(0.985, 0.995),
    scale=0.95,
)


# ============================================================
# 14. Resource legend in bottom-right, vertical, boxed
# ============================================================
resource_handles = []
resource_labels = []

for key in ["energy", "downlink", "queue"]:
    face = RESOURCE_STYLES[key]["color"]
    edge = darken(face, 0.62)

    patch = plt.Rectangle(
        (0, 0),
        1,
        1,
        facecolor=face,
        edgecolor=edge,
        linewidth=0.85,
        hatch=RESOURCE_STYLES[key]["hatch"],
    )
    resource_handles.append(patch)
    resource_labels.append(RESOURCE_STYLES[key]["label"])

resource_legend = ax_br.legend(
    resource_handles,
    resource_labels,
    loc="upper right",
    bbox_to_anchor=(0.985, 0.985),
    ncol=1,
    frameon=True,
    fontsize=RESOURCE_LEGEND_SIZE,
    title="Resources",
    title_fontsize=RESOURCE_LEGEND_TITLE_SIZE,
    handlelength=1.5,
    handletextpad=0.45,
    labelspacing=0.42,
    borderpad=0.42,
    borderaxespad=0.20,
)

resource_legend.get_frame().set_edgecolor("#4D4D4D")
resource_legend.get_frame().set_linewidth(0.9)
resource_legend.get_frame().set_facecolor("white")
resource_legend.get_frame().set_alpha(1.0)
resource_legend.set_zorder(20)


# ============================================================
# 15. Remove grids from legend boxes visually
# ============================================================
# (The legend frame is opaque white; no extra action is needed.)


# ============================================================
# 16. Row labels only: (a) and (b), centered below each row
# ============================================================
fig.text(
    0.50,
    0.505,
    "(a)",
    ha="center",
    va="center",
    fontsize=PANEL_LABEL_SIZE,
)

fig.text(
    0.50,
    0.035,
    "(b)",
    ha="center",
    va="center",
    fontsize=PANEL_LABEL_SIZE,
)


# ============================================================
# 17. Export normalized values for record
# ============================================================
csv_path = OUTPUT_DIR / "normalized_sensitivity_ablation_values.csv"
rows = [
    ["group", "x", "metric", "normalized_mean_percent", "normalized_std_percent"],
]

for i, xv in enumerate(capacity_x):
    for key in ["J_all", "J_HV", "N_HV"]:
        rows.append([
            "capacity",
            xv,
            key,
            capacity_norm[key][i],
            capacity_err[key][i],
        ])

for i, xv in enumerate(value_labels):
    for key in ["J_all", "J_HV", "N_HV"]:
        rows.append([
            "value_profile",
            xv,
            key,
            value_norm[key][i],
            value_err[key][i],
        ])

for i, xv in enumerate(ablation_labels):
    for key in ["J_all", "J_HV", "N_HV"]:
        rows.append([
            "ablation_performance",
            xv,
            key,
            ablation_norm[key][i],
            ablation_err[key][i],
        ])

for i, xv in enumerate(ablation_labels):
    for key in ["energy", "downlink", "queue"]:
        rows.append([
            "ablation_resource",
            xv,
            key,
            resource_norm[key][i],
            resource_err[key][i],
        ])

np.savetxt(
    csv_path,
    np.asarray(rows, dtype=object),
    fmt="%s",
    delimiter=",",
)


# ============================================================
# 18. Export figures
# ============================================================
pdf_path = OUTPUT_DIR / "sensitivity_ablation_2x2_zoom_transparent_source_box.pdf"
png_path = OUTPUT_DIR / "sensitivity_ablation_2x2_zoom_transparent_source_box.png"

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

fig.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight",
    facecolor="white",
)

print(f"Vector PDF saved to: {pdf_path.resolve()}")
print(f"High-resolution PNG saved to: {png_path.resolve()}")
print(f"Normalized values saved to: {csv_path.resolve()}")

if SHOW_FIGURE:
    plt.show()
else:
    plt.close(fig)
