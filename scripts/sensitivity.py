from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.ticker import MultipleLocator


# ============================================================
# 1. Output settings
# ============================================================
OUTPUT_DIR = Path("compact_sensitivity_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SHOW_FIGURE = True


# ============================================================
# 2. Reuse the palette from the previous result figure
# ============================================================
PALETTE = {
    "myopic_blue": "#B2E6FD",
    "b1_green": "#B8D2CC",
    "offload_orange": "#E8B2A7",
    "tut_yellow": "#FEEBB9",
    "prop_purple": "#B5AED5",
}

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


def darken(hex_color: str, factor: float = 0.68) -> tuple[float, float, float]:
    """Create a darker edge/line color from the original pastel palette."""
    rgb = np.array(mcolors.to_rgb(hex_color))
    return tuple(np.clip(rgb * factor, 0.0, 1.0))


# ============================================================
# 3. PROP sensitivity data: mean ± standard deviation
#    Main reference scenario: V25_CAP050
# ============================================================

# (a-left) Capacity sensitivity under the main value profile V25
capacity_x = np.array([0.25, 0.35, 0.50, 0.65])

capacity_mean = {
    "J_all": np.array([70.417, 80.643, 85.401, 88.322]),
    "J_HV": np.array([44.814, 55.672, 60.444, 63.502]),
    "N_HV": np.array([39.2, 42.8, 44.6, 46.2]),
}

capacity_std = {
    "J_all": np.array([3.455, 1.671, 2.294, 1.227]),
    "J_HV": np.array([4.517, 2.918, 3.817, 1.824]),
    "N_HV": np.array([2.049, 2.168, 2.074, 2.168]),
}

# (a-right) Urgent-value sensitivity under capacity = 0.50
urgent_value_x = np.array([10, 25, 50])

value_mean = {
    "J_all": np.array([84.370, 85.401, 85.123]),
    "J_HV": np.array([49.766, 60.444, 64.666]),
    "N_HV": np.array([44.2, 44.6, 44.8]),
}

value_std = {
    "J_all": np.array([1.293, 2.294, 2.553]),
    "J_HV": np.array([2.106, 3.817, 4.114]),
    "N_HV": np.array([1.304, 2.074, 1.789]),
}


# ============================================================
# 4. Normalize all metrics to the main setting V25_CAP050 = 100%
#    Error bars are normalized by the same reference mean.
# ============================================================
REFERENCE_MEAN = {
    "J_all": capacity_mean["J_all"][2],
    "J_HV": capacity_mean["J_HV"][2],
    "N_HV": capacity_mean["N_HV"][2],
}


def normalize(
    means: dict[str, np.ndarray],
    stds: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    normalized_mean = {}
    normalized_std = {}

    for key in means:
        ref = REFERENCE_MEAN[key]
        normalized_mean[key] = means[key] / ref * 100.0
        normalized_std[key] = stds[key] / ref * 100.0

    return normalized_mean, normalized_std


capacity_norm, capacity_err = normalize(capacity_mean, capacity_std)
value_norm, value_err = normalize(value_mean, value_std)


# ============================================================
# 5. Publication font settings
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# 6. Compact two-panel sensitivity row
#    Designed for a double-column Magazine figure.
#    Later, the ablation panel can be placed underneath as panel (b).
# ============================================================
fig, axes = plt.subplots(
    1,
    2,
    figsize=(7.25, 2.85),
    sharey=True,
    facecolor="white",
)

fig.subplots_adjust(
    left=0.105,
    right=0.985,
    bottom=0.23,
    top=0.78,
    wspace=0.18,
)


def draw_sensitivity_panel(
    ax,
    x_values,
    means,
    errors,
    x_label,
    x_ticks,
    main_x,
):
    # Main-setting reference
    ax.axhline(
        100,
        color="#888888",
        linestyle=(0, (3, 2)),
        linewidth=0.9,
        zorder=1,
    )

    ax.axvline(
        main_x,
        color="#B5AED5",
        linestyle=(0, (2, 2)),
        linewidth=0.9,
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
            markersize=5.8,
            markerfacecolor=face,
            markeredgecolor=edge,
            markeredgewidth=0.9,
            linewidth=1.45,
            elinewidth=0.9,
            capsize=2.5,
            capthick=0.9,
            label=style["label"],
            zorder=3,
        )

    ax.set_xlabel(x_label, fontsize=9.5, labelpad=4)
    ax.set_xticks(x_ticks)
    ax.tick_params(axis="both", labelsize=8.4, width=0.9, length=3.5)

    ax.set_ylim(68, 111)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    ax.grid(
        axis="y",
        which="major",
        linestyle=":",
        linewidth=0.55,
        alpha=0.45,
        zorder=0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Mark the adopted main setting without a large annotation box
    ax.text(
        main_x,
        108.4,
        "Main",
        ha="center",
        va="center",
        fontsize=7.7,
        color=darken(PALETTE["prop_purple"], 0.58),
    )


draw_sensitivity_panel(
    axes[0],
    capacity_x,
    capacity_norm,
    capacity_err,
    "Downlink capacity multiplier",
    capacity_x,
    0.50,
)

draw_sensitivity_panel(
    axes[1],
    urgent_value_x,
    value_norm,
    value_err,
    r"Urgent-task value $V_{\mathrm{urgent}}$",
    urgent_value_x,
    25,
)

axes[0].set_ylabel(
    "Relative performance (%)",
    fontsize=9.5,
    labelpad=5,
)

# Small internal descriptors, not full subplot titles
axes[0].text(
    0.02,
    0.97,
    "Capacity variation",
    transform=axes[0].transAxes,
    ha="left",
    va="top",
    fontsize=9.1,
)

axes[1].text(
    0.02,
    0.97,
    "Value heterogeneity",
    transform=axes[1].transAxes,
    ha="left",
    va="top",
    fontsize=9.1,
)

# Whole sensitivity block label; this can remain panel (a)
fig.text(
    0.105,
    0.925,
    "(a) Sensitivity analysis",
    ha="left",
    va="center",
    fontsize=10.3,
    fontweight="bold",
)

# Shared legend above both panels
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.63, 0.915),
    ncol=3,
    frameon=False,
    fontsize=8.7,
    handlelength=1.7,
    handletextpad=0.35,
    columnspacing=1.05,
)

# ============================================================
# 7. Export
# ============================================================
pdf_path = OUTPUT_DIR / "sensitivity_analysis_compact.pdf"
png_path = OUTPUT_DIR / "sensitivity_analysis_compact.png"
csv_path = OUTPUT_DIR / "normalized_sensitivity_values.csv"

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

# Save normalized values for later ablation/figure integration
rows = [
    ["group", "x", "metric", "normalized_mean_percent", "normalized_std_percent"],
]

for x_value_index, x_value in enumerate(capacity_x):
    for key in METRIC_STYLES:
        rows.append([
            "capacity",
            x_value,
            key,
            capacity_norm[key][x_value_index],
            capacity_err[key][x_value_index],
        ])

for x_value_index, x_value in enumerate(urgent_value_x):
    for key in METRIC_STYLES:
        rows.append([
            "urgent_value",
            x_value,
            key,
            value_norm[key][x_value_index],
            value_err[key][x_value_index],
        ])

np.savetxt(
    csv_path,
    np.asarray(rows, dtype=object),
    fmt="%s",
    delimiter=",",
)

print(f"Vector PDF saved to: {pdf_path.resolve()}")
print(f"High-resolution PNG saved to: {png_path.resolve()}")
print(f"Normalized values saved to: {csv_path.resolve()}")

if SHOW_FIGURE:
    plt.show()
else:
    plt.close(fig)