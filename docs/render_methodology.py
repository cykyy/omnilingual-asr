"""Render the methodology overview flowchart to PNG using matplotlib.

Run:
    python docs/render_methodology.py

Produces docs/methodology_overview.png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent / "methodology_overview.png"

# Canvas
FIG_W, FIG_H = 14.0, 17.5
DPI = 200

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.set_aspect("equal")
ax.axis("off")

# Title
ax.text(
    FIG_W / 2,
    FIG_H - 0.35,
    "Omnilingual ASR — Methodology Overview",
    ha="center",
    va="top",
    fontsize=20,
    fontweight="bold",
    color="#111",
)


def draw_node(
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    lines: list[str],
    fill: str,
    edge: str,
    text_color: str = "#111",
    title_size: int = 12,
    body_size: int = 10,
    bold_title: bool = True,
    dashed: bool = False,
) -> tuple[float, float, float, float]:
    """Draw a rounded rectangle node with title + body lines. Returns (x, y, w, h) bbox."""
    ls = (0, (4, 3)) if dashed else "solid"
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.6,
        facecolor=fill,
        edgecolor=edge,
        linestyle=ls,
    )
    ax.add_patch(box)
    # Title
    ax.text(
        x,
        y + h / 2 - 0.24,
        title,
        ha="center",
        va="top",
        fontsize=title_size,
        fontweight="bold" if bold_title else "normal",
        color=text_color,
    )
    # Body lines
    if lines:
        line_h = 0.24
        total = len(lines) * line_h
        start_y = y + h / 2 - 0.55
        for i, line in enumerate(lines):
            ax.text(
                x,
                start_y - i * line_h,
                line,
                ha="center",
                va="top",
                fontsize=body_size,
                color=text_color,
            )
    return x - w / 2, y - h / 2, w, h


def arrow(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str = "#333",
    dashed: bool = False,
    label: str | None = None,
    lw: float = 1.4,
    head: bool = True,
) -> None:
    ls = (0, (4, 3)) if dashed else "solid"
    style = "-|>" if head else "-"
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        mutation_scale=14,
        color=color,
        linewidth=lw,
        linestyle=ls,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arr)
    if label:
        ax.text(
            (x1 + x2) / 2 + 0.1,
            (y1 + y2) / 2,
            label,
            fontsize=9,
            color="#666",
            style="italic",
            ha="left",
            va="center",
        )


# Color palette per stage
COLORS = {
    "data":      ("#DCE8F7", "#335B8C"),
    "prep":      ("#E6F0E4", "#4F7A48"),
    "tok":       ("#FFF3D6", "#B38B2F"),
    "model":     ("#FDE2CE", "#B45C2E"),
    "train":     ("#E8DDF5", "#6B3FA0"),
    "decode":    ("#D9EEF2", "#2F7A88"),
    "eval":      ("#F7D6E0", "#A93A5C"),
    "analysis":  ("#F2E6D9", "#8C5A2A"),
    "findings":  ("#FBD1C7", "#B23A2A"),
    "hardware":  ("#E8E8E8", "#666666"),
}

SPINE_X = 5.2
NODE_W = 6.8
NODE_H_BIG = 1.35
NODE_H_SM = 0.95

# Vertical layout (top to bottom)
spine = [
    ("data", "1 · DATA", [
        "Bengali regional speech  (~80 hr)",
        "16 kHz · mono · TSV/WRD manifests",
        "RegSpeech12 · 12 dialects · 17,049 clips",
    ], 15.2, NODE_H_BIG),
    ("prep", "2 · PREPROCESSING", [
        "AudioDecoder → 16 kHz resample",
        "Mono mixdown · LayerNorm",
        "No SpecAugment or augmentation",
    ], 13.5, NODE_H_BIG),
    ("tok", "3 · TOKENIZATION", [
        "Character-level",
        "omniASR_tokenizer_written_v2",
        "No text normalization",
    ], 11.8, NODE_H_BIG),
    ("model", "4 · MODEL", [
        "Wav2Vec2 encoder  (CNN + Transformer, 320×↓)",
        "→  CTC head  (omniASR_CTC_300M_v2, 325M params)",
        "Pretrained → fine-tuned",
    ], 10.1, NODE_H_BIG),
    ("train", "5 · TRAINING", [
        "CTC loss · AdamW (β=0.9/0.98, wd=0.01)",
        "bfloat16 · length batching · grad. accum. 8",
        "Warmup 10% → Hold 40% → Decay 50% · 10K steps",
    ], 8.4, NODE_H_BIG),
    ("decode", "6 · DECODING", [
        "Greedy CTC  ·  no language model",
    ], 6.9, NODE_H_SM),
    ("eval", "7 · EVALUATION", [
        "WER · UER (strict, test split)",
        "per-rank  ref.txt / hyp.txt",
    ], 5.6, 1.1),
    ("analysis", "8 · ANALYSIS", [], 4.25, 0.75),
]

spine_centers: dict[str, tuple[float, float, float]] = {}  # key -> (x, y, h)
for key, title, body, y, h in spine:
    fill, edge = COLORS[key]
    draw_node(SPINE_X, y, NODE_W, h, title, body, fill, edge)
    spine_centers[key] = (SPINE_X, y, h)

# Spine arrows
order = ["data", "prep", "tok", "model", "train", "decode", "eval", "analysis"]
for a, b in zip(order, order[1:]):
    xa, ya, ha = spine_centers[a]
    xb, yb, hb = spine_centers[b]
    arrow(xa, ya - ha / 2, xb, yb + hb / 2, color="#333", lw=1.6)

# Analysis fan-out leaves
leaves = [
    ("Per-dialect WER / CER", 1.25),
    ("Error types\n(Sub / Del / Ins)", 3.55),
    ("Script confusion\n(Bengali ↔ Assamese)", 5.85),
    ("Vocabulary coverage", 8.15),
]
LEAF_Y = 2.9
LEAF_W = 2.15
LEAF_H = 0.85
fill, edge = COLORS["analysis"]
leaf_centers: list[tuple[float, float]] = []
for text, lx in leaves:
    box = FancyBboxPatch(
        (lx - LEAF_W / 2, LEAF_Y - LEAF_H / 2),
        LEAF_W,
        LEAF_H,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.4,
        facecolor=fill,
        edgecolor=edge,
    )
    ax.add_patch(box)
    ax.text(
        lx,
        LEAF_Y,
        text,
        ha="center",
        va="center",
        fontsize=9.5,
        color="#3A2410",
    )
    leaf_centers.append((lx, LEAF_Y))

# Arrows from 8·ANALYSIS to each leaf
ax_x, ax_y, ax_h = spine_centers["analysis"]
for lx, ly in leaf_centers:
    arrow(ax_x, ax_y - ax_h / 2, lx, ly + LEAF_H / 2, color="#555", lw=1.2)

# 9 · FINDINGS node
F_Y = 1.15
F_H = 1.2
fill, edge = COLORS["findings"]
draw_node(
    SPINE_X,
    F_Y,
    NODE_W,
    F_H,
    "9 · FINDINGS",
    [
        "Zero-shot baseline:   97.93% WER",
        "Fine-tuned (10K steps, 80 hr):   72.97% WER",
    ],
    fill,
    edge,
    title_size=13,
)

# Arrows from leaves into findings
for lx, ly in leaf_centers:
    arrow(lx, ly - LEAF_H / 2, SPINE_X, F_Y + F_H / 2, color="#555", lw=1.1)

# Hardware side-rail (right side, alongside training)
HW_X = 11.35
fill, edge = COLORS["hardware"]
# Cluster label
ax.text(
    HW_X,
    10.95,
    "Hardware / Stack",
    ha="center",
    va="bottom",
    fontsize=11,
    fontweight="bold",
    color="#333",
)
# Dashed cluster border
cluster = FancyBboxPatch(
    (HW_X - 1.55, 7.55),
    3.1,
    3.4,
    boxstyle="round,pad=0.05,rounding_size=0.12",
    linewidth=1.3,
    facecolor="#F7F7F7",
    edgecolor="#888",
    linestyle=(0, (5, 3)),
)
ax.add_patch(cluster)
# Two hardware nodes
draw_node(
    HW_X, 9.85, 2.7, 0.85,
    "1 GPU (16 GB)",
    ["bfloat16"],
    fill, edge,
    title_size=11, body_size=9.5, dashed=True,
)
draw_node(
    HW_X, 8.3, 2.7, 0.9,
    "fairseq2",
    ["PyTorch · torchaudio"],
    fill, edge,
    title_size=11, body_size=9.5, dashed=True,
)

# Dashed "supports" edges from hardware cluster to stages 4–6
supports_label_done = False
for stage_key in ("model", "train", "decode"):
    sx, sy, sh = spine_centers[stage_key]
    arrow(
        HW_X - 1.55,
        sy,
        sx + NODE_W / 2,
        sy,
        color="#888",
        dashed=True,
        lw=1.1,
        head=False,
        label="supports" if not supports_label_done else None,
    )
    supports_label_done = True

# Footer caption
ax.text(
    FIG_W / 2,
    0.15,
    "Data → Preprocessing → Tokenizer → Model → Training → Decoding → Evaluation → Analysis → Findings",
    ha="center",
    va="bottom",
    fontsize=9,
    color="#666",
    style="italic",
)

plt.tight_layout(pad=0.3)
plt.savefig(OUT, dpi=DPI, bbox_inches="tight", facecolor="white")
print(f"Wrote {OUT}  ({OUT.stat().st_size / 1024:.1f} KB)")
