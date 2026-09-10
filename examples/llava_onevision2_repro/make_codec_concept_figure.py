#!/usr/bin/env python3
"""Generates codec_canvas_concept.png: a step-by-step schematic of
cv-preinfer's frame selection, grounded in what the checkpoint's own
codec_video_processing_llava_onevision2.py confirms (not guessed):

  1) uniformly sample 512 candidate frames, split into 16 groups of 32
     (`num_sampled_frames() = (target_canvas//images_per_group)*group_size`)
  2) score each group's 32 frames by bit-cost/motion-vector "readiness",
     keep the best 4 -- each kept frame becomes its OWN canvas (NOT a
     multi-frame packed collage: `drop_padding_canvases` treats each
     canvas as having one uniform timestamp across all its patches,
     confirming one frame per canvas)
  3) repeat for all 16 groups -> 4 canvases/group x 16 groups = 64
     canvases total -> fed to the VLM
     (`expected_canv = (usable_frames//group_size)*images_per_group`
     confirms the per-group canvas count directly)

Uses the default config (target_canvas=64, group_size=32,
images_per_group=4).

No GPU/model needed -- pure illustration from already-documented
mechanics (see SMOKE_TEST.md).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GREY = "#c9c8c2"
GRID = "#e3e2dd"


def box(ax, x, y, w, h, text, color, textcolor="white", fontsize=8.5):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.06",
                           linewidth=0, facecolor=color, zorder=3)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=textcolor, fontweight="bold", zorder=4)


def arrow(ax, x0, y0, x1, y1):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11,
                         linewidth=1.5, color=TEXT_SECONDARY, zorder=3)
    ax.add_patch(a)


def main():
    fig = plt.figure(figsize=(10.5, 7.2), facecolor=SURFACE)

    # ---- Panel 1: whole video -> 16 groups of 32 = 512 candidates ----
    ax1 = fig.add_axes((0.04, 0.72, 0.92, 0.24))
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 2.2)
    ax1.axis("off")
    ax1.text(0.0, 2.0, "1) Uniformly sample 512 candidate frames, split into 16 groups of 32",
             fontsize=10.5, fontweight="bold", color=TEXT_PRIMARY, ha="left")

    n_groups = 16
    gx0, gx1 = 0.1, 9.9
    gw = (gx1 - gx0) / n_groups
    for i in range(n_groups):
        x = gx0 + i * gw
        color = ORANGE if i == 5 else BLUE
        ax1.add_patch(Rectangle((x, 0.5), gw * 0.88, 0.9, facecolor=color,
                                 edgecolor=SURFACE, linewidth=1.2, zorder=2))
    ax1.text((gx0 + gx1) / 2, 0.15,
             "16 groups × 32 frames each = 512 candidate frames total (spread evenly across the video)",
             ha="center", fontsize=7.5, color=TEXT_SECONDARY)
    ax1.text(gx0 + 5 * gw + gw * 0.44, 0.3, "zoom in ↓", ha="center", fontsize=6.5,
              color=TEXT_SECONDARY, style="italic")

    # ---- Panel 2: zoom into one group of 32 -> keep best 4 ----
    ax2 = fig.add_axes((0.04, 0.40, 0.55, 0.26))
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 2.6)
    ax2.axis("off")
    ax2.text(0.0, 2.4, '2) Score all 32 frames in a group by "readiness",',
             fontsize=9, fontweight="bold", color=TEXT_PRIMARY, ha="left")
    ax2.text(0.0, 2.1, "keep the best 4 (bit-cost + motion-vector signal)",
             fontsize=9, fontweight="bold", color=TEXT_PRIMARY, ha="left")

    kept_idx = {3, 11, 19, 27}
    cols = 8
    cw, ch = 1.15, 0.42
    for i in range(32):
        r, c = divmod(i, cols)
        x = 0.1 + c * cw
        y = 1.7 - r * (ch + 0.06)
        color = ORANGE if i in kept_idx else GREY
        ax2.add_patch(Rectangle((x, y), cw * 0.86, ch, facecolor=color, edgecolor=SURFACE, linewidth=1, zorder=2))
    ax2.add_patch(Rectangle((0.05, 1.65 - 3 * (ch + 0.06) - 0.05), cols * cw + 0.05, 4 * (ch + 0.06) + 0.15,
                             fill=False, edgecolor=GRID, linewidth=1, zorder=1))
    legend_y = 0.15
    ax2.add_patch(Rectangle((0.1, legend_y), 0.3, 0.22, facecolor=ORANGE, zorder=2))
    ax2.text(0.55, legend_y + 0.11, "kept (top 4 readiness score)", fontsize=7.3, color=TEXT_SECONDARY, va="center")
    ax2.add_patch(Rectangle((4.3, legend_y), 0.3, 0.22, facecolor=GREY, zorder=2))
    ax2.text(4.75, legend_y + 0.11, "discarded (28 of 32)", fontsize=7.3, color=TEXT_SECONDARY, va="center")

    arrow(ax2, 9.3, 1.0, 9.9, 1.0)

    # ---- Panel 3: each of the 4 kept frames becomes its OWN canvas
    # (confirmed: drop_padding_canvases treats each canvas as having one
    # uniform timestamp across all its patches -- one frame per canvas,
    # not a multi-frame packed collage).
    ax3 = fig.add_axes((0.62, 0.40, 0.34, 0.26))
    ax3.set_xlim(0, 5)
    ax3.set_ylim(0, 2.6)
    ax3.axis("off")
    ax3.text(0.0, 2.5, "3) Each kept frame", fontsize=8.7, fontweight="bold", color=TEXT_PRIMARY, ha="left", va="top")
    ax3.text(0.0, 2.25, "becomes its own canvas", fontsize=8.7, fontweight="bold", color=TEXT_PRIMARY, ha="left", va="top")
    cs = 0.55
    canvas_positions = [(0.3, 1.4), (1.15, 1.4), (2.0, 1.4), (2.85, 1.4)]
    for x, y in canvas_positions:
        ax3.add_patch(Rectangle((x, y), cs, cs, fill=False, edgecolor=AQUA, linewidth=1.8, zorder=1))
        ax3.add_patch(Rectangle((x + 0.08, y + 0.08), cs - 0.16, cs - 0.16, facecolor=ORANGE, edgecolor=SURFACE, linewidth=1, zorder=2))
    ax3.text(1.7, 0.9, "4 canvases,\none frame each", ha="center", fontsize=7.5, color=TEXT_SECONDARY)

    # ---- Panel 4: repeat x16 -> 64 canvases -> VLM ----
    ax4 = fig.add_axes((0.04, 0.06, 0.92, 0.26))
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 2.2)
    ax4.axis("off")
    ax4.text(0.0, 2.0, "4) Repeat for all 16 groups → 64 canvases total → fed to the VLM",
             fontsize=10.5, fontweight="bold", color=TEXT_PRIMARY, ha="left")

    n_canvas_groups = 16
    cx0, cx1 = 0.1, 6.6
    cw2 = (cx1 - cx0) / n_canvas_groups
    for i in range(n_canvas_groups):
        x = cx0 + i * cw2
        ax4.add_patch(Rectangle((x, 0.6), cw2 * 0.82, 0.8, facecolor=AQUA, edgecolor=SURFACE, linewidth=1, zorder=2))
    ax4.text((cx0 + cx1) / 2, 0.2, "16 groups × 4 canvases each = 64 canvases",
             ha="center", fontsize=7.5, color=TEXT_SECONDARY)

    arrow(ax4, 6.9, 1.0, 7.5, 1.0)
    box(ax4, 7.6, 0.55, 2.3, 0.9, "fed to VLM\n(llava_onevision2)", TEXT_PRIMARY, fontsize=8.5)

    fig.suptitle('What "512→64 canvases" actually means: from candidates to selected frames',
                 x=0.02, y=0.995, ha="left", fontsize=13, fontweight="bold", color=TEXT_PRIMARY)

    out_png = Path(__file__).parent / "codec_canvas_concept.png"
    fig.savefig(out_png, dpi=170, facecolor=SURFACE)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
