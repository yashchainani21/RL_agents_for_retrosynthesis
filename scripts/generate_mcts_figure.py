#!/usr/bin/env python3
"""
Generate a publication-quality conceptual figure illustrating
how Monte Carlo Tree Search (MCTS) explores a retrosynthesis tree.

Shows the four MCTS phases (Selection, Expansion, Rollout, Backpropagation)
annotated on an intentionally asymmetric tree, contrasted with a BFS inset.

Usage:
    python scripts/generate_mcts_figure.py [--output PATH] [--dpi N]

Outputs:
    figures/mcts_tree_conceptual.png  (raster, 300 DPI default)
    figures/mcts_tree_conceptual.pdf  (vector)
    figures/mcts_tree_conceptual.svg  (vector)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Color palette (consistent with DORAnet_agent/visualize.py)
# ---------------------------------------------------------------------------
COLOR = {
    "root": "#f39c12",
    "internal": "#3498db",
    "terminal": "#2ecc71",
    "selection_path": "#1a5276",
    "backprop": "#e74c3c",
    "unexpanded": "#95a5a6",
    "edge": "#bdc3c7",
    "bfs": "#aab7b8",
    "new_node_fill": "#d6eaf8",
    "new_node_edge": "#e67e22",
    "rollout": "#7f8c8d",
    "bg": "#ffffff",
    "phase_text": "#2c3e50",
}


# ---------------------------------------------------------------------------
# 1. Build synthetic asymmetric MCTS tree
# ---------------------------------------------------------------------------

def _build_tree() -> Tuple[nx.DiGraph, Dict]:
    """
    Return (G, node_meta) for a hand-crafted asymmetric MCTS tree.

    node_meta[node_id] = {
        "visits": int,
        "label": str,          # display label
        "kind": str,            # "root" | "internal" | "terminal" | "unexpanded" | "new"
    }
    """
    G = nx.DiGraph()
    meta: Dict[str, dict] = {}

    def _add(nid, visits, kind="internal", label=None):
        G.add_node(nid)
        meta[nid] = {
            "visits": visits,
            "label": label or nid,
            "kind": kind,
        }

    # Root
    _add("Root", 50, kind="root")

    # --- Left / deep branch (heavily explored) ---
    _add("A", 28)
    G.add_edge("Root", "A")

    _add("A1", 18)
    G.add_edge("A", "A1")

    _add("A2", 8)
    G.add_edge("A", "A2")

    _add("A1a", 12)
    G.add_edge("A1", "A1a")

    _add("A1b", 4)
    G.add_edge("A1", "A1b")

    _add("A1a-i", 8)
    G.add_edge("A1a", "A1a-i")

    _add("A1a-ii", 3)
    G.add_edge("A1a", "A1a-ii")

    _add("T1", 5, kind="terminal", label="T1")  # terminal / building block
    G.add_edge("A1a-i", "T1")

    _add("NEW", 0, kind="new", label="NEW")  # just expanded this iteration
    G.add_edge("A1a-i", "NEW")

    _add("A1a-iii", 0, kind="unexpanded")
    G.add_edge("A1a-ii", "A1a-iii")

    _add("A2a", 5)
    G.add_edge("A2", "A2a")

    _add("A2b", 0, kind="unexpanded")
    G.add_edge("A2", "A2b")

    # --- Middle branch (moderate exploration) ---
    _add("B", 12)
    G.add_edge("Root", "B")

    _add("B1", 7)
    G.add_edge("B", "B1")

    _add("B2", 4, kind="internal")
    G.add_edge("B", "B2")

    _add("B1a", 3)
    G.add_edge("B1", "B1a")

    _add("B1b", 2)
    G.add_edge("B1", "B1b")

    # --- Right / shallow branches (barely explored) ---
    _add("C", 5)
    G.add_edge("Root", "C")

    _add("C1", 2)
    G.add_edge("C", "C1")

    _add("C2", 0, kind="unexpanded")
    G.add_edge("C", "C2")

    _add("D", 2)
    G.add_edge("Root", "D")

    _add("D1", 0, kind="unexpanded")
    G.add_edge("D", "D1")

    return G, meta


# ---------------------------------------------------------------------------
# 2. Leaf-count-weighted hierarchical layout
# ---------------------------------------------------------------------------

def _leaf_count(G: nx.DiGraph, node) -> int:
    children = list(G.successors(node))
    if not children:
        return 1
    return sum(_leaf_count(G, c) for c in children)


def _hierarchical_pos(
    G: nx.DiGraph,
    root,
    x_range: Tuple[float, float] = (-5.0, 5.0),
    y_step: float = 1.4,
) -> Dict:
    """Leaf-count-weighted top-down layout."""
    pos: Dict = {}
    total_leaves = _leaf_count(G, root)
    total_width = x_range[1] - x_range[0]

    def _layout(node, x_lo, x_hi, depth):
        pos[node] = ((x_lo + x_hi) / 2.0, -depth * y_step)
        children = list(G.successors(node))
        if not children:
            return
        child_leaves = [_leaf_count(G, c) for c in children]
        total = sum(child_leaves)
        cursor = x_lo
        for child, cl in zip(children, child_leaves):
            share = (cl / total) * (x_hi - x_lo)
            _layout(child, cursor, cursor + share, depth + 1)
            cursor += share

    _layout(root, x_range[0], x_range[1], 0)
    return pos


# ---------------------------------------------------------------------------
# 3. Node drawing helpers
# ---------------------------------------------------------------------------

def _node_alpha(visits: int) -> float:
    if visits >= 10:
        return 1.0
    if visits >= 3:
        return 0.7
    if visits >= 1:
        return 0.45
    return 0.25


def _node_radius(visits: int) -> float:
    if visits >= 10:
        return 0.32
    if visits >= 3:
        return 0.25
    if visits >= 1:
        return 0.20
    return 0.17


def _draw_node(ax, x, y, meta, node_id, zorder=3):
    kind = meta["kind"]
    visits = meta["visits"]
    alpha = _node_alpha(visits)
    r = _node_radius(visits)

    if kind == "root":
        circle = plt.Circle(
            (x, y), r, fc=COLOR["root"], ec="#2c3e50",
            lw=2.2, alpha=alpha, zorder=zorder,
        )
        ax.add_patch(circle)
    elif kind == "terminal":
        half = r * 0.85
        rect = FancyBboxPatch(
            (x - half, y - half), 2 * half, 2 * half,
            boxstyle="round,pad=0.04",
            fc=COLOR["terminal"], ec="#27ae60",
            lw=2.2, alpha=alpha, zorder=zorder,
        )
        ax.add_patch(rect)
    elif kind == "unexpanded":
        circle = plt.Circle(
            (x, y), r, fc="white", ec=COLOR["unexpanded"],
            lw=1.5, ls="--", alpha=0.8, zorder=zorder,
        )
        ax.add_patch(circle)
    elif kind == "new":
        circle = plt.Circle(
            (x, y), r, fc=COLOR["new_node_fill"], ec=COLOR["new_node_edge"],
            lw=2.0, ls="--", alpha=0.9, zorder=zorder,
        )
        ax.add_patch(circle)
    else:  # internal
        circle = plt.Circle(
            (x, y), r, fc=COLOR["internal"], ec="#2c3e50",
            lw=1.5, alpha=alpha, zorder=zorder,
        )
        ax.add_patch(circle)

    # Node label (n=X) below the node
    label_text = f"n={visits}"
    # Shift label left for selection path nodes so it doesn't overlap backprop arrows
    label_x = x
    label_ha = "center"
    if node_id in ("Root", "A", "A1", "A1a", "A1a-i"):
        label_x = x - r - 0.05
        label_ha = "right"
    ax.text(
        label_x, y - r - 0.12, label_text,
        ha=label_ha, va="top", fontsize=6.5,
        color="#2c3e50", zorder=zorder + 1,
    )


# ---------------------------------------------------------------------------
# 4. Draw edges
# ---------------------------------------------------------------------------

def _draw_edges(ax, G, pos, meta, selection_path_edges):
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        r_u = _node_radius(meta[u]["visits"])
        r_v = _node_radius(meta[v]["visits"])

        # Shorten line to avoid overlapping circles
        dx, dy = x1 - x0, y1 - y0
        length = np.hypot(dx, dy)
        if length == 0:
            continue
        ux_, uy_ = dx / length, dy / length
        sx, sy = x0 + ux_ * r_u, y0 + uy_ * r_u
        ex, ey = x1 - ux_ * r_v, y1 - uy_ * r_v

        if (u, v) in selection_path_edges:
            # Selection path: bold dark blue
            ax.plot(
                [sx, ex], [sy, ey],
                color=COLOR["selection_path"], lw=3.0, solid_capstyle="round",
                zorder=2,
            )
        else:
            ax.plot(
                [sx, ex], [sy, ey],
                color=COLOR["edge"], lw=1.3, solid_capstyle="round",
                zorder=1,
            )


# ---------------------------------------------------------------------------
# 5. Annotate MCTS phases
# ---------------------------------------------------------------------------

def _annotate_phases(ax, pos, meta):
    # --- Phase 1: Selection ---
    selection_nodes = ["Root", "A", "A1", "A1a", "A1a-i"]
    # Place label to the left of the selection path
    sx, sy = pos["A1"][0] - 1.15, (pos["Root"][1] + pos["A1a"][1]) / 2
    ax.text(
        sx, sy,
        "1  Selection",
        fontsize=11, fontweight="bold", color=COLOR["selection_path"],
        ha="right", va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLOR["selection_path"],
                  alpha=0.9, lw=1.2),
        zorder=10,
    )
    # Small curved arrow from label to path
    mid_x, mid_y = pos["A1"][0], pos["A1"][1]
    ax.annotate(
        "", xy=(mid_x - _node_radius(meta["A1"]["visits"]) - 0.05, mid_y),
        xytext=(sx + 0.05, sy),
        arrowprops=dict(arrowstyle="-|>", color=COLOR["selection_path"],
                        lw=1.0, connectionstyle="arc3,rad=0.15"),
        zorder=10,
    )

    # --- Phase 2: Expansion ---
    nx_, ny = pos["NEW"]
    ax.text(
        nx_ + 0.6, ny + 0.15,
        "2  Expansion",
        fontsize=11, fontweight="bold", color=COLOR["new_node_edge"],
        ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLOR["new_node_edge"],
                  alpha=0.9, lw=1.2),
        zorder=10,
    )
    ax.annotate(
        "", xy=(nx_ + _node_radius(0) + 0.05, ny),
        xytext=(nx_ + 0.55, ny + 0.12),
        arrowprops=dict(arrowstyle="-|>", color=COLOR["new_node_edge"],
                        lw=1.0, connectionstyle="arc3,rad=-0.2"),
        zorder=10,
    )

    # --- Phase 3: Rollout (simulation) ---
    rx, ry = nx_, ny - _node_radius(0)
    # Dashed arrow downward from NEW node
    rollout_end_y = ry - 0.9
    ax.annotate(
        "",
        xy=(rx, rollout_end_y),
        xytext=(rx, ry - 0.1),
        arrowprops=dict(arrowstyle="-|>", color=COLOR["rollout"],
                        lw=1.5, ls="--"),
        zorder=5,
    )
    ax.text(
        rx + 0.45, rollout_end_y + 0.3,
        "3  Rollout",
        fontsize=11, fontweight="bold", color=COLOR["rollout"],
        ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLOR["rollout"],
                  alpha=0.9, lw=1.2),
        zorder=10,
    )
    # "reward = ?" label at the end of the dashed arrow
    ax.text(
        rx, rollout_end_y - 0.15,
        "reward?",
        fontsize=8, fontstyle="italic", color=COLOR["rollout"],
        ha="center", va="top",
        zorder=10,
    )

    # --- Phase 4: Backpropagation ---
    # Red upward arrows alongside the selection path
    backprop_nodes = ["A1a-i", "A1a", "A1", "A", "Root"]
    offset_x = 0.35  # horizontal offset from the node centers
    for i in range(len(backprop_nodes) - 1):
        lower = backprop_nodes[i]
        upper = backprop_nodes[i + 1]
        lx, ly = pos[lower]
        ux_, uy_ = pos[upper]
        r_lower = _node_radius(meta[lower]["visits"])
        r_upper = _node_radius(meta[upper]["visits"])
        ax.annotate(
            "",
            xy=(ux_ + offset_x, uy_ - r_upper - 0.05),
            xytext=(lx + offset_x, ly + r_lower + 0.05),
            arrowprops=dict(arrowstyle="-|>", color=COLOR["backprop"],
                            lw=1.8, connectionstyle="arc3,rad=0.12"),
            zorder=6,
        )
        # "+r" label next to each arrow
        mid_x_bp = (lx + ux_) / 2 + offset_x + 0.18
        mid_y_bp = (ly + uy_) / 2
        ax.text(
            mid_x_bp, mid_y_bp, "+r",
            fontsize=7, color=COLOR["backprop"], fontstyle="italic",
            ha="left", va="center", zorder=7,
        )

    # Backprop label near top-right of the root
    bx, by = pos["Root"][0] + offset_x + 0.8, pos["Root"][1] + 0.6
    ax.text(
        bx, by,
        "4  Backpropagation",
        fontsize=11, fontweight="bold", color=COLOR["backprop"],
        ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=COLOR["backprop"],
                  alpha=0.9, lw=1.2),
        zorder=10,
    )
    ax.annotate(
        "",
        xy=(pos["Root"][0] + _node_radius(meta["Root"]["visits"]) + offset_x + 0.02,
            pos["Root"][1] + 0.05),
        xytext=(bx - 0.05, by - 0.15),
        arrowprops=dict(arrowstyle="-|>", color=COLOR["backprop"],
                        lw=1.0, connectionstyle="arc3,rad=-0.15"),
        zorder=10,
    )


# ---------------------------------------------------------------------------
# 6. BFS comparison inset
# ---------------------------------------------------------------------------

def _draw_bfs_inset(fig, rect=(0.64, 0.62, 0.34, 0.34)):
    """
    Draw a small complete binary tree in an inset axes to contrast with MCTS.
    rect = [left, bottom, width, height] in figure coords.
    """
    ax_inset = fig.add_axes(rect)
    ax_inset.set_xlim(-2.3, 2.3)
    ax_inset.set_ylim(-3.5, 0.8)
    ax_inset.set_aspect("equal")
    ax_inset.axis("off")

    # Build complete binary tree depth 3
    bfs_g = nx.balanced_tree(2, 3, create_using=nx.DiGraph())
    bfs_pos = _hierarchical_pos(bfs_g, 0, x_range=(-2.0, 2.0), y_step=1.0)

    # Draw edges
    for u, v in bfs_g.edges():
        x0, y0 = bfs_pos[u]
        x1, y1 = bfs_pos[v]
        ax_inset.plot([x0, x1], [y0, y1], color=COLOR["bfs"], lw=1.0, zorder=1)

    # Draw nodes — all same size, same shade
    r = 0.16
    for n in bfs_g.nodes():
        x, y = bfs_pos[n]
        c = plt.Circle((x, y), r, fc=COLOR["bfs"], ec="#7f8c8d", lw=1.0, zorder=2)
        ax_inset.add_patch(c)

    # Title
    ax_inset.text(
        0, 0.55,
        "Breadth-First Search",
        fontsize=9, fontweight="bold", ha="center", va="bottom",
        color="#566573",
    )
    ax_inset.text(
        0, 0.25,
        "uniform exploration",
        fontsize=7.5, fontstyle="italic", ha="center", va="bottom",
        color="#7f8c8d",
    )

    # Thin border for the inset
    for spine in ax_inset.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#bdc3c7")
        spine.set_linewidth(0.8)


# ---------------------------------------------------------------------------
# 7. Legend
# ---------------------------------------------------------------------------

def _draw_legend(ax):
    legend_elements = [
        plt.Circle((0, 0), 0.1, fc=COLOR["internal"], ec="#2c3e50", lw=1.5),
    ]
    # Use Line2D proxies for legend
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR["root"],
               markeredgecolor="#2c3e50", markersize=10, lw=0,
               label="Root (target molecule)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR["internal"],
               markeredgecolor="#2c3e50", markersize=10, lw=0,
               label="Explored node (darker = more visits)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
               markeredgecolor=COLOR["unexpanded"], markersize=10, lw=0,
               markeredgewidth=1.5, label="Unexpanded node"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR["terminal"],
               markeredgecolor="#27ae60", markersize=10, lw=0,
               label="Building block (terminal)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR["new_node_fill"],
               markeredgecolor=COLOR["new_node_edge"], markersize=10, lw=0,
               markeredgewidth=1.5, label="Newly expanded node"),
        Line2D([0], [0], color=COLOR["selection_path"], lw=2.5,
               label="Selection path (UCB1)"),
        Line2D([0], [0], color=COLOR["backprop"], lw=2.0,
               marker=">", markersize=5,
               label="Backpropagation (+reward)"),
    ]
    ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=8,
        framealpha=0.92,
        edgecolor="#bdc3c7",
        fancybox=True,
        borderpad=0.8,
        handletextpad=0.6,
    )


# ---------------------------------------------------------------------------
# Main figure assembly
# ---------------------------------------------------------------------------

def generate_figure(output_dir: str = "figures", dpi: int = 300) -> None:
    G, meta = _build_tree()
    pos = _hierarchical_pos(G, "Root", x_range=(-5.5, 5.5), y_step=1.5)

    selection_path_edges = {
        ("Root", "A"), ("A", "A1"), ("A1", "A1a"),
        ("A1a", "A1a-i"), ("A1a-i", "NEW"),
    }

    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Draw edges first (lowest z-order)
    _draw_edges(ax, G, pos, meta, selection_path_edges)

    # Draw nodes
    for nid in G.nodes():
        x, y = pos[nid]
        _draw_node(ax, x, y, meta[nid], nid)

    # Annotate MCTS phases
    _annotate_phases(ax, pos, meta)

    # Title
    ax.set_title(
        "Monte Carlo Tree Search for Retrosynthesis",
        fontsize=15, fontweight="bold", color=COLOR["phase_text"],
        pad=16,
    )
    ax.text(
        0.5, 1.01,
        "Asymmetric exploration guided by UCB1 selection policy",
        transform=ax.transAxes, fontsize=10, fontstyle="italic",
        ha="center", va="bottom", color="#7f8c8d",
    )

    # Axis limits with padding
    xs = [pos[n][0] for n in G.nodes()]
    ys = [pos[n][1] for n in G.nodes()]
    pad_x, pad_y = 1.6, 1.8
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y - 0.8, max(ys) + pad_y)
    ax.set_aspect("equal")
    ax.axis("off")

    # BFS inset
    _draw_bfs_inset(fig)

    # Legend
    _draw_legend(ax)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02)

    # Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stem = out / "mcts_tree_conceptual"
    for fmt in ("png", "pdf", "svg"):
        path = stem.with_suffix(f".{fmt}")
        save_kwargs = dict(
            bbox_inches="tight", facecolor="white", edgecolor="none",
        )
        if fmt == "png":
            save_kwargs["dpi"] = dpi
        fig.savefig(path, format=fmt, **save_kwargs)
        print(f"Saved {path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a publication-quality MCTS conceptual figure.",
    )
    parser.add_argument(
        "--output", default="figures",
        help="Output directory (default: figures/)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="DPI for PNG output (default: 300)",
    )
    args = parser.parse_args()
    generate_figure(output_dir=args.output, dpi=args.dpi)


if __name__ == "__main__":
    main()
