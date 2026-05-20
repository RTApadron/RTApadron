"""Well mechanical state schematic generator.

Creates a publication-quality matplotlib cross-section figure from a
WellMechConfig.  The figure updates on each call — pass it directly to
st.pyplot() in Streamlit or save with fig.savefig().

Coordinate system
-----------------
X: radial position in inches from centerline (0 = center, + = right side).
   Symmetric left/right representation (full cross-section view).
Y: measured depth in ft, 0 at surface (top of plot), increasing downward.
   matplotlib y-axis is inverted so surface appears at the top.
"""

from __future__ import annotations

import math
import io

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle, FancyArrow

from src.services.well_mech_qc_service import WellMechConfig, MechQCResult

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

_C_FORMATION    = "#D9CEB2"   # light tan — formation
_C_OPEN_HOLE    = "#C8B99A"   # slightly darker tan — open hole below shoe
_C_CEMENT       = "#B0B0B0"   # gray — cement behind casings
_C_FLUID        = "#DBEAFE"   # pale blue — wellbore fluid / mud
_C_CASING       = ["#3A3A3A", "#5A5A5A", "#7A7A7A", "#9A9A9A"]  # dark → light per string
_C_TUBING       = "#1D4ED8"   # blue
_C_TUBING_FLUID = "#93C5FD"   # light blue — fluid inside tubing
_C_PERF         = "#DC2626"   # red — perforations
_C_ESP          = "#D97706"   # amber — ESP
_C_SURFACE      = "#374151"   # dark gray — surface formation cap
_C_CENTERLINE   = "#9CA3AF"   # light gray

# minimum visual wall thickness in inches (prevents tiny walls from disappearing)
_MIN_VISUAL_WALL_IN = 0.12


def _visual_wall(c_od: float, c_id: float) -> float:
    """Ensure steel wall is at least _MIN_VISUAL_WALL_IN wide for drawing."""
    real = (c_od - c_id) / 2.0
    return max(real, _MIN_VISUAL_WALL_IN)


def draw_well_schematic(
    config: WellMechConfig,
    qc_results: list[MechQCResult] | None = None,
    figsize: tuple[float, float] = (7.5, 13),
) -> plt.Figure:
    """Draw a symmetric cross-section well schematic.

    Args:
        config:     Well mechanical configuration.
        qc_results: Optional list of QC results — used to colour-code the
                    depth annotations (not yet implemented; reserved).
        figsize:    (width, height) in inches for the figure.

    Returns:
        matplotlib Figure object.
    """
    td = config.effective_total_depth
    max_od = max((c.od_in for c in config.casings), default=config.tubing.od_in)
    x_max = max_od / 2.0 + 2.5   # margin beyond outermost OD for labels

    fig, ax = plt.subplots(figsize=figsize, facecolor="#FAFAFA")
    ax.set_facecolor(_C_FORMATION)
    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(td + td * 0.06, -td * 0.04)   # surface at top, small margin

    # ── Surface cap (ground level) ─────────────────────────────────────────
    ax.add_patch(Rectangle(
        (-x_max, -td * 0.04), 2 * x_max, td * 0.04,
        facecolor=_C_SURFACE, edgecolor="none", zorder=20,
    ))
    ax.text(0, -td * 0.02, "Superficie (0 ft)",
            ha="center", va="center", fontsize=8, fontweight="bold",
            color="white", zorder=21)

    # ── Wellbore fluid background (inside borehole to TD) ──────────────────
    bore_half_id = (config.casings[-1].id_in / 2.0) if config.casings else x_max * 0.5
    ax.add_patch(Rectangle(
        (-bore_half_id, 0), 2 * bore_half_id, td,
        facecolor=_C_FLUID, edgecolor="none", alpha=0.5, zorder=1,
    ))

    # ── Open hole below innermost casing shoe ──────────────────────────────
    if config.casings:
        inner = config.casings[-1]
        shoe_inner = inner.shoe_depth_ft
        hole_half = inner.od_in / 2.0 + 0.15   # open hole slightly wider than casing OD
        if shoe_inner < td:
            ax.add_patch(Rectangle(
                (-hole_half, shoe_inner), 2 * hole_half, td - shoe_inner,
                facecolor=_C_OPEN_HOLE, edgecolor="#8B7355",
                linewidth=0.7, linestyle="--", zorder=2,
            ))

    # ── Cement strip behind outermost casing ──────────────────────────────
    if config.casings:
        outer = config.casings[0]
        outer_half_od = outer.od_in / 2.0
        cement_w = 0.35   # cement column width in inches (visual only)
        for sign in (-1, 1):
            ax.add_patch(Rectangle(
                (sign * outer_half_od, 0), sign * cement_w, outer.shoe_depth_ft,
                facecolor=_C_CEMENT, edgecolor="none",
                alpha=0.6, zorder=1, hatch="///",
            ))

    # ── Casing strings (outermost to innermost) ────────────────────────────
    for idx, casing in enumerate(config.casings):
        color = _C_CASING[min(idx, len(_C_CASING) - 1)]
        half_od = casing.od_in / 2.0
        vw = _visual_wall(casing.od_in, casing.id_in)
        half_id = half_od - vw        # adjusted for visual clarity

        shoe = casing.shoe_depth_ft
        zord = 3 + idx

        for sign in (-1, 1):
            # steel wall
            ax.add_patch(Rectangle(
                (sign * half_id, 0), sign * vw, shoe,
                facecolor=color, edgecolor="#111111",
                linewidth=0.4, zorder=zord,
            ))

        # casing shoe (horizontal bar)
        ax.plot([-half_od, half_od], [shoe, shoe],
                color=color, linewidth=2.5, zorder=zord + 0.5, solid_capstyle="butt")

        # ── Right annotation ──
        label_x = x_max * 0.52
        ax.annotate(
            f"{casing.name}\n{casing.od_in:.3f}\" OD × {casing.id_in:.3f}\" ID\nzapato: {shoe:.0f} ft",
            xy=(half_od, shoe * 0.45),
            xytext=(label_x, shoe * 0.45),
            fontsize=6.5, color=color, va="center",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.5, shrinkA=0, shrinkB=0),
        )

    # ── Tubing ─────────────────────────────────────────────────────────────
    t_od = config.tubing.od_in
    t_id = config.tubing.id_in
    t_set = config.tubing.set_depth_ft
    t_half_od = t_od / 2.0
    t_vw = _visual_wall(t_od, t_id)
    t_half_id = t_half_od - t_vw

    # Fluid inside tubing
    ax.add_patch(Rectangle(
        (-t_half_id, 0), 2 * t_half_id, t_set,
        facecolor=_C_TUBING_FLUID, edgecolor="none", alpha=0.7, zorder=7,
    ))

    for sign in (-1, 1):
        ax.add_patch(Rectangle(
            (sign * t_half_id, 0), sign * t_vw, t_set,
            facecolor=_C_TUBING, edgecolor="#1E3A8A",
            linewidth=0.5, zorder=8,
        ))

    # tubing shoe
    ax.plot([-t_half_od, t_half_od], [t_set, t_set],
            color=_C_TUBING, linewidth=2.5, zorder=9, solid_capstyle="butt")

    # left annotation for tubing
    label_x_left = -x_max * 0.52
    ax.annotate(
        f"Tubing\n{t_od:.3f}\" OD × {t_id:.3f}\" ID\nzapato: {t_set:.0f} ft",
        xy=(-t_half_od, t_set * 0.5),
        xytext=(label_x_left, t_set * 0.5),
        fontsize=6.5, color=_C_TUBING, va="center", ha="right",
        arrowprops=dict(arrowstyle="-", color=_C_TUBING, lw=0.5, shrinkA=0, shrinkB=0),
    )

    # ── ESP ────────────────────────────────────────────────────────────────
    if config.has_esp and config.esp_intake_depth_ft is not None:
        esp_d = config.esp_intake_depth_ft
        esp_h = max(100, td * 0.025)   # visual height proportional to TD
        esp_w = t_half_id * 0.80
        ax.add_patch(Rectangle(
            (-esp_w, esp_d - esp_h), 2 * esp_w, esp_h,
            facecolor=_C_ESP, edgecolor="#92400E",
            linewidth=1.2, zorder=10,
        ))
        ax.text(0, esp_d - esp_h / 2, "ESP",
                ha="center", va="center", fontsize=7, fontweight="bold",
                color="#1C0600", zorder=11)
        # annotation right
        ax.annotate(
            f"Intake ESP\n{esp_d:.0f} ft",
            xy=(esp_w, esp_d),
            xytext=(x_max * 0.52, esp_d + td * 0.025),
            fontsize=6.5, color=_C_ESP,
            arrowprops=dict(arrowstyle="->", color=_C_ESP, lw=0.7),
        )

    # ── Perforations ───────────────────────────────────────────────────────
    p_top = config.perfs_top_ft
    p_bot = config.perfs_bottom_ft
    if p_bot > p_top:
        inner = config.innermost_casing
        if inner:
            p_half_od = inner.od_in / 2.0
            p_vw = _visual_wall(inner.od_in, inner.id_in)
            p_half_id = p_half_od - p_vw

            # red hatch on casing walls at perf interval
            for sign in (-1, 1):
                ax.add_patch(Rectangle(
                    (sign * p_half_id, p_top), sign * p_vw, p_bot - p_top,
                    facecolor=_C_PERF, edgecolor="none",
                    alpha=0.9, zorder=12,
                ))

            # perforation shots (horizontal lines shooting outward)
            n_shots = max(4, min(12, int((p_bot - p_top) / 60)))
            for i in range(n_shots):
                y_shot = p_top + (i + 0.5) * (p_bot - p_top) / n_shots
                for sign in (-1, 1):
                    ax.annotate(
                        "",
                        xy=(sign * (p_half_od + 0.4), y_shot),
                        xytext=(sign * p_half_id, y_shot),
                        arrowprops=dict(
                            arrowstyle="-|>", color=_C_PERF,
                            lw=0.9, mutation_scale=5,
                        ),
                        zorder=13,
                    )

        # perf depth bracket (left side)
        bracket_x = -x_max * 0.52
        ax.annotate(
            "",
            xy=(bracket_x + 0.3, p_top),
            xytext=(bracket_x + 0.3, p_bot),
            arrowprops=dict(arrowstyle="<->", color=_C_PERF, lw=1.2),
        )
        ax.text(bracket_x, (p_top + p_bot) / 2,
                f"Perforaciones\n{p_top:.0f}–{p_bot:.0f} ft\n({p_bot - p_top:.0f} ft)",
                ha="right", va="center", fontsize=6.5, color=_C_PERF, zorder=14)

    # ── Centerline ─────────────────────────────────────────────────────────
    ax.axvline(0, color=_C_CENTERLINE, linewidth=0.6, linestyle="--", zorder=0, alpha=0.7)

    # ── Depth reference grid ───────────────────────────────────────────────
    depth_marks = set()
    for c in config.casings:
        depth_marks.add(c.shoe_depth_ft)
    depth_marks.add(config.tubing.set_depth_ft)
    depth_marks.add(config.perfs_top_ft)
    depth_marks.add(config.perfs_bottom_ft)
    if config.has_esp and config.esp_intake_depth_ft:
        depth_marks.add(config.esp_intake_depth_ft)

    for d in depth_marks:
        ax.axhline(d, color="#CCCCCC", linewidth=0.35, linestyle=":", zorder=0, alpha=0.6)

    # ── Axes formatting ────────────────────────────────────────────────────
    ax.set_ylabel("Profundidad MD (ft)", fontsize=9, labelpad=8)
    ax.set_xlabel("Diámetro (pulgadas)", fontsize=8)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    )
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f'{abs(v):.2f}"')
    )
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(axis="y", color="#BBBBBB", linewidth=0.25, alpha=0.4)

    # ── Legend ─────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(color=_C_CASING[min(i, len(_C_CASING) - 1)], label=c.name)
        for i, c in enumerate(config.casings)
    ] + [
        mpatches.Patch(color=_C_TUBING, label=f"Tubing {config.tubing.od_in:.3f}\" OD"),
        mpatches.Patch(color=_C_PERF, label="Perforaciones"),
    ]
    if config.has_esp and config.esp_intake_depth_ft:
        handles.append(mpatches.Patch(color=_C_ESP, label="ESP"))
    ax.legend(handles=handles, loc="lower left", fontsize=6.5, framealpha=0.85)

    # ── Title ──────────────────────────────────────────────────────────────
    ax.set_title(
        f"Estado Mecánico — {config.well_id}\n"
        f"Perfs: {config.perfs_top_ft:.0f}–{config.perfs_bottom_ft:.0f} ft MD",
        fontsize=10, fontweight="bold", pad=12,
    )

    fig.tight_layout()
    return fig


def schematic_to_png_bytes(config: WellMechConfig, dpi: int = 150) -> bytes:
    """Return schematic as PNG bytes (for st.download_button)."""
    fig = draw_well_schematic(config)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
