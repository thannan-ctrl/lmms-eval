#!/usr/bin/env python3
"""Generates codec_canvas_concept.png: an academic-figure-style schematic
of the frame-level selection this repo's `codec` backend actually runs
(NOT the OneVision-Encoder paper's patch-level codec-aligned sparsity --
that's a different mechanism; this repo's codec backend uses a separate,
older tool literally named "legacy-exact"). Grounded in what the
checkpoint's own codec_video_processing_llava_onevision2.py confirms:

  (a) uniformly sample 512 candidate frames, split into 16 groups of 32
      (`num_sampled_frames() = (target_canvas//images_per_group)*group_size`)
  (b) score every frame in a group by codec "readiness" (bit-cost +
      motion-vector signal read from the H264 bitstream)
  (c) keep the top images_per_group=4 per group, discard the rest
  (d) each kept frame becomes its own canvas (one frame per canvas,
      confirmed by drop_padding_canvases' per-canvas uniform-timestamp
      check) -> 16 groups x 4 canvases = 64 canvases -> fed to the VLM

No GPU/model needed -- pure illustration from already-documented
mechanics (see SMOKE_TEST.md).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import ConnectionPatch, FancyBboxPatch, Rectangle  # noqa: E402

SURFACE = "#ffffff"
PANEL_BG = "#fafaf9"
TEXT_PRIMARY = "#111111"
TEXT_SECONDARY = "#5a5a58"
BLUE = "#2a78d6"
ORANGE = "#d9481f"
GREY = "#c9c8c2"
BORDER = "#d8d7d2"

# Warm-to-cool "readiness score" ramp (low -> high), academic saliency-map
# convention: cool/blue = low signal, warm/red = high signal.
SCORE_RAMP = ["#dbe7f5", "#a9c7e8", "#f4c28a", "#e8863f", "#c1421a"]


def score_color(v: float) -> str:
    """v in [0,1] -> a hex color from SCORE_RAMP."""
    n = len(SCORE_RAMP) - 1
    idx = min(n, int(v * (n + 1)))
    return SCORE_RAMP[idx]


def panel_frame(ax):
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                            edgecolor=BORDER, linewidth=1, zorder=0))


def label(ax, text):
    ax.text(0.02, -0.07, text, transform=ax.transAxes, fontsize=10,
             fontweight="bold", color=TEXT_PRIMARY, ha="left", va="top")


def main():
    fig = plt.figure(figsize=(12.5, 3.6), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 4, left=0.03, right=0.985, top=0.86, bottom=0.20, wspace=0.12)

    # ---------------------------------------------------------- (a)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_facecolor(PANEL_BG)
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)
    ax_a.axis("off")
    panel_frame(ax_a)
    ax_a.set_title("Uniform temporal sampling", fontsize=9.5, color=TEXT_PRIMARY, pad=6)

    n_groups = 16
    x0, x1 = 0.06, 0.94
    gw = (x1 - x0) / n_groups
    highlight = 6
    for i in range(n_groups):
        x = x0 + i * gw
        c = ORANGE if i == highlight else BLUE
        ax_a.add_patch(Rectangle((x, 0.35), gw * 0.8, 0.34, facecolor=c,
                                  edgecolor="white", linewidth=0.8, zorder=2))
    ax_a.text(0.5, 0.16, "512 candidates,\n16 groups × 32 frames",
              ha="center", va="top", fontsize=7.8, color=TEXT_SECONDARY)
    ax_a.annotate("", xy=(0.06, 0.85), xytext=(0.94, 0.85),
                  arrowprops=dict(arrowstyle="-", color=TEXT_SECONDARY, lw=0.8))
    ax_a.text(0.5, 0.88, "source video (temporal axis)", ha="center", fontsize=7.3,
              color=TEXT_SECONDARY, style="italic")
    label(ax_a, "a")

    # ---------------------------------------------------------- (b)
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_facecolor(PANEL_BG)
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    ax_b.axis("off")
    panel_frame(ax_b)
    ax_b.set_title('Per-frame "readiness" score', fontsize=9.5, color=TEXT_PRIMARY, pad=6)

    scores = [
        0.12, 0.30, 0.18, 0.95, 0.22, 0.10, 0.15, 0.28,
        0.20, 0.35, 0.14, 0.88, 0.19, 0.24, 0.11, 0.31,
        0.16, 0.27, 0.13, 0.92, 0.21, 0.17, 0.09, 0.33,
        0.23, 0.29, 0.12, 0.97, 0.18, 0.26, 0.14, 0.20,
    ]
    cols, rows = 8, 4
    m = 0.06
    cw = (1 - 2 * m) / cols
    ch = 0.5 / rows
    top = 0.82
    for i, s in enumerate(scores):
        r, c = divmod(i, cols)
        x = m + c * cw
        y = top - (r + 1) * ch
        ax_b.add_patch(Rectangle((x, y), cw * 0.86, ch * 0.8, facecolor=score_color(s),
                                  edgecolor="white", linewidth=0.6, zorder=2))
    ax_b.text(0.5, 0.16,
              "bit-cost + motion vectors,\nread from the H264 bitstream",
              ha="center", va="top", fontsize=7.8, color=TEXT_SECONDARY)
    # small colorbar-like legend
    lx0, ly = 0.08, 0.86
    for j, col in enumerate(SCORE_RAMP):
        ax_b.add_patch(Rectangle((lx0 + j * 0.045, ly), 0.04, 0.035, facecolor=col,
                                  edgecolor="none", zorder=2))
    ax_b.text(lx0, ly + 0.05, "low", fontsize=6.6, color=TEXT_SECONDARY, ha="left")
    ax_b.text(lx0 + 0.045 * (len(SCORE_RAMP) - 1) + 0.04, ly + 0.05, "high",
              fontsize=6.6, color=TEXT_SECONDARY, ha="right")
    label(ax_b, "b")

    # ---------------------------------------------------------- (c)
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.set_facecolor(PANEL_BG)
    ax_c.set_xlim(0, 1)
    ax_c.set_ylim(0, 1)
    ax_c.axis("off")
    panel_frame(ax_c)
    ax_c.set_title("Keep top-4, discard the rest", fontsize=9.5, color=TEXT_PRIMARY, pad=6)

    kept_idx = {3, 11, 19, 27}
    for i in range(32):
        r, c = divmod(i, cols)
        x = m + c * cw
        y = top - (r + 1) * ch
        col = ORANGE if i in kept_idx else GREY
        alpha = 1.0 if i in kept_idx else 0.55
        ax_c.add_patch(Rectangle((x, y), cw * 0.86, ch * 0.8, facecolor=col,
                                  edgecolor="white", linewidth=0.6, zorder=2, alpha=alpha))
    leg_y = 0.16
    ax_c.add_patch(Rectangle((0.08, leg_y), 0.035, 0.035, facecolor=ORANGE, zorder=2))
    ax_c.text(0.13, leg_y + 0.018, "kept (4 of 32)", fontsize=7.3, color=TEXT_SECONDARY, va="center")
    ax_c.add_patch(Rectangle((0.55, leg_y), 0.035, 0.035, facecolor=GREY, alpha=0.55, zorder=2))
    ax_c.text(0.6, leg_y + 0.018, "discarded", fontsize=7.3, color=TEXT_SECONDARY, va="center")
    label(ax_c, "c")

    # ---------------------------------------------------------- (d)
    ax_d = fig.add_subplot(gs[0, 3])
    ax_d.set_facecolor(PANEL_BG)
    ax_d.set_xlim(0, 1)
    ax_d.set_ylim(0, 1)
    ax_d.axis("off")
    panel_frame(ax_d)
    ax_d.set_title("Kept frames → canvases → VLM", fontsize=9.5, color=TEXT_PRIMARY, pad=6)

    # one group's 4 kept frames as individual canvases (small)
    cs = 0.11
    xs = [0.10, 0.24, 0.38, 0.52]
    for x in xs:
        ax_d.add_patch(Rectangle((x, 0.58), cs, cs, fill=False, edgecolor=ORANGE, linewidth=1.3, zorder=2))
        ax_d.add_patch(Rectangle((x + 0.018, 0.598), cs - 0.036, cs - 0.036,
                                  facecolor=ORANGE, alpha=0.55, edgecolor="none", zorder=2))
    ax_d.text(0.35, 0.50, "4 canvases from this group", ha="center", fontsize=7, color=TEXT_SECONDARY)
    ax_d.annotate("", xy=(0.68, 0.635), xytext=(0.60, 0.635),
                  arrowprops=dict(arrowstyle="->", color=TEXT_SECONDARY, lw=1.0))
    ax_d.text(0.85, 0.64, "×16\ngroups", ha="center", va="center", fontsize=7.3,
              color=TEXT_SECONDARY, fontweight="bold")

    ax_d.add_patch(FancyBboxPatch((0.06, 0.10), 0.88, 0.22, boxstyle="round,pad=0.01,rounding_size=0.03",
                                   linewidth=0, facecolor=TEXT_PRIMARY, zorder=3))
    ax_d.text(0.5, 0.21, "64 canvases → VLM", ha="center", va="center", fontsize=8.3,
              color="white", fontweight="bold", zorder=4)
    label(ax_d, "d")

    # Connect panels a->b->c->d with thin arrows across the figure
    for src_ax, dst_ax in [(ax_a, ax_b), (ax_b, ax_c), (ax_c, ax_d)]:
        con = ConnectionPatch(
            xyA=(1.0, 0.5), coordsA=src_ax.transAxes,
            xyB=(0.0, 0.5), coordsB=dst_ax.transAxes,
            arrowstyle="-|>", mutation_scale=14, linewidth=1.3, color=TEXT_PRIMARY,
        )
        fig.add_artist(con)

    fig.suptitle(
        "Frame-level codec-aligned selection (this repo's `codec` backend)",
        x=0.03, y=0.975, ha="left", fontsize=13, fontweight="bold", color=TEXT_PRIMARY,
    )
    fig.text(
        0.03, 0.02,
        "Figure. cv-preinfer (codec-video-prep-legacy-exact) drops whole frames using H264 -- distinct from the\n"
        "OneVision-Encoder paper's method (arXiv:2602.08683, github.com/EvolvingLMMs-Lab/OneVision-Encoder), which\n"
        "sparsifies patches within HEVC frames (I-frames kept whole) and shares no code/terminology with this tool.",
        fontsize=7.6, color=TEXT_SECONDARY, ha="left", va="bottom",
    )

    out_png = Path(__file__).parent / "codec_canvas_concept.png"
    fig.savefig(out_png, dpi=200, facecolor=SURFACE)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
