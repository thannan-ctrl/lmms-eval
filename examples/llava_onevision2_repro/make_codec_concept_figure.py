#!/usr/bin/env python3
"""Generates codec_canvas_concept.png: a schematic of how the codec
backend's cv-preinfer tool goes from a video to its 64 output canvases
(512 candidate frames -> readiness/bit-cost scoring -> keep the best 64).
No GPU/model needed -- pure illustration from already-documented
mechanics (see SMOKE_TEST.md).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GRID = "#e3e2dd"


def box(ax, x, y, w, h, text, color, textcolor="white", fontsize=8.5):
    rect = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0, facecolor=color, zorder=2,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=textcolor, fontweight="bold", zorder=3)


def arrow(ax, x0, y0, x1, y1, label=None):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                         linewidth=1.6, color=TEXT_SECONDARY, zorder=2)
    ax.add_patch(a)
    if label:
        ax.text((x0 + x1) / 2, y0 + 0.28, label, ha="center", va="bottom",
                fontsize=6.8, color=TEXT_SECONDARY, style="italic")


def main():
    fig, ax = plt.subplots(figsize=(9.5, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # Timeline strip representing the source video.
    ax.add_patch(FancyBboxPatch((0.15, 3.55), 2.1, 0.5, boxstyle="round,pad=0.01,rounding_size=0.05",
                                 linewidth=1, edgecolor=GRID, facecolor="white", zorder=1))
    for i in range(10):
        ax.plot([0.3 + i * 0.2, 0.3 + i * 0.2], [3.6, 4.0], color=GRID, linewidth=1, zorder=1)
    ax.text(1.2, 4.28, "source video", ha="center", fontsize=8, color=TEXT_SECONDARY)

    arrow(ax, 2.3, 3.8, 2.85, 3.8, "uniform\nsample")
    box(ax, 2.9, 3.45, 1.7, 0.85, "512 candidate\nframes", BLUE, fontsize=8)

    arrow(ax, 4.65, 3.8, 5.2, 3.8, "16 groups\nof 32")
    box(ax, 5.25, 3.45, 1.9, 0.85, "cv-preinfer:\nreadiness / bit-cost\nscoring", ORANGE, fontsize=7.8)
    ax.text(6.2, 3.1, "reads H264 motion vectors +\nbits-per-block directly from\nthe compressed stream",
            ha="center", fontsize=6.3, color=TEXT_SECONDARY, style="italic")

    arrow(ax, 7.15, 3.8, 7.7, 3.8, "keep best 4\nper group")
    box(ax, 7.75, 3.45, 1.9, 0.85, "64 canvases\n(packed images)", AQUA, fontsize=8)

    ax.add_patch(FancyArrowPatch((8.7, 3.35), (8.72, 2.5), arrowstyle="-|>", mutation_scale=12,
                                  linewidth=1.6, color=TEXT_SECONDARY, zorder=2))
    box(ax, 6.9, 1.7, 3.65, 0.8, "fed to the VLM (llava_onevision2)", TEXT_PRIMARY, fontsize=8.5)

    ax.text(
        5, 0.55,
        "Why H264/HEVC only: bit-cost/motion-vector signal is a byproduct of\n"
        "how those codecs compress video -- cv-preinfer reads it straight from the\n"
        "bitstream instead of re-deriving it via full pixel-level analysis (\"codec-aligned sparsity\").",
        ha="center", va="center", fontsize=7.3, color=TEXT_SECONDARY,
    )

    fig.suptitle("How codec picks its 64 canvases from a video", x=0.02, y=0.98, ha="left",
                 fontsize=12, fontweight="bold", color=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0.02, 1, 0.9))

    out_png = Path(__file__).parent / "codec_canvas_concept.png"
    fig.savefig(out_png, dpi=170, facecolor=SURFACE)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
