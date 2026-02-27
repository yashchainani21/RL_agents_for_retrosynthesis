"""
Visualization module for DORAnet MCTS search trees.

Creates interactive and static visualizations showing:
- Tree structure with nodes and edges
- PKS library matches highlighted
- Node statistics (visits, value, iteration)
- Retrosynthetic pathways
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
import math

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

if TYPE_CHECKING:
    from .mcts import DORAnetMCTS
    from .node import Node


def create_tree_graph(agent: "DORAnetMCTS") -> nx.DiGraph:
    """
    Create a NetworkX directed graph from the DORAnet MCTS tree.

    Args:
        agent: The DORAnetMCTS agent after running.

    Returns:
        NetworkX DiGraph with node attributes.
    """
    G = nx.DiGraph()

    # Add nodes with attributes
    for node in agent.nodes:
        is_pks_match = agent._is_in_pks_library(node.smiles or "")
        avg_value = node.value / node.visits if node.visits > 0 else 0

        # Truncate SMILES for display
        smiles_short = node.smiles[:20] + "..." if node.smiles and len(node.smiles) > 20 else node.smiles

        G.add_node(
            node.node_id,
            smiles=node.smiles,
            smiles_short=smiles_short,
            depth=node.depth,
            visits=node.visits,
            value=node.value,
            avg_value=avg_value,
            provenance=node.provenance or "target",
            is_pks_match=is_pks_match,
            is_sink_compound=node.is_sink_compound,
            sink_compound_type=node.sink_compound_type,
            expanded=node.expanded,
            created_at=node.created_at_iteration,
            expanded_at=node.expanded_at_iteration,
            selected_at=node.selected_at_iterations,
        )

    # Add edges
    for parent_id, child_id in agent.edges:
        G.add_edge(parent_id, child_id)

    return G


def get_hierarchical_pos(G: nx.DiGraph, root: int = 0) -> Dict[int, Tuple[float, float]]:
    """
    Create a hierarchical layout for the tree.

    Args:
        G: NetworkX DiGraph
        root: Root node ID

    Returns:
        Dictionary mapping node IDs to (x, y) positions.
    """
    pos = {}

    def _hierarchy_pos(node, x=0, y=0, layer=1, width=1.0, dx=1.0):
        pos[node] = (x, -y)  # Negative y so tree grows downward
        children = list(G.successors(node))
        if children:
            # Distribute children evenly
            child_width = width / len(children)
            next_x = x - width / 2 + child_width / 2
            for child in children:
                _hierarchy_pos(child, next_x, y + dy, layer + 1, child_width, dx)
                next_x += child_width

    # Calculate vertical spacing
    max_depth = max((G.nodes[n].get('depth', 0) for n in G.nodes), default=0)
    dy = 1.0 if max_depth == 0 else 2.0 / max(max_depth, 1)

    # Calculate initial width based on number of leaf nodes
    leaves = [n for n in G.nodes if G.out_degree(n) == 0]
    width = max(len(leaves) * 0.5, 2.0)

    _hierarchy_pos(root, x=0, y=0, width=width)

    return pos


def visualize_doranet_tree(
    agent: "DORAnetMCTS",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10),
    show_smiles: bool = True,
    show_stats: bool = True,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Create a visualization of the DORAnet MCTS search tree.

    Args:
        agent: The DORAnetMCTS agent after running.
        output_path: Path to save the figure (optional).
        figsize: Figure size in inches.
        show_smiles: Whether to show SMILES labels on nodes.
        show_stats: Whether to show visit/value statistics.
        title: Custom title for the plot.

    Returns:
        Matplotlib Figure object.
    """
    from rdkit import Chem

    # Create graph
    G = create_tree_graph(agent)

    if len(G.nodes) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No nodes to visualize", ha='center', va='center', fontsize=14)
        return fig

    # Get hierarchical layout
    pos = get_hierarchical_pos(G, root=0)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Separate nodes into regular nodes and sink compounds (for different shapes)
    regular_nodes = []
    sink_nodes = []

    regular_colors = []
    regular_sizes = []
    regular_edge_colors = []

    sink_colors = []
    sink_sizes = []
    sink_edge_colors = []

    for node_id in G.nodes:
        node_data = G.nodes[node_id]
        is_pks = node_data.get('is_pks_match', False)
        is_sink = node_data.get('is_sink_compound', False)
        provenance = node_data.get('provenance', 'target')
        visits = node_data.get('visits', 1)

        # Size based on visits (min 300, max 2000)
        size = min(300 + visits * 100, 2000)

        # Determine color based on provenance (same for both sink and non-sink)
        if is_pks and not is_sink:
            color = '#2ecc71'  # Green for PKS match (non-sink)
            edge_color = '#27ae60'
        elif provenance == 'enzymatic':
            color = '#3498db'  # Blue for enzymatic
            edge_color = '#2c3e50'
        elif provenance == 'synthetic':
            color = '#9b59b6'  # Purple for synthetic
            edge_color = '#2c3e50'
        else:
            color = '#f39c12'  # Orange for target
            edge_color = '#2c3e50'

        if is_sink:
            # Sink compounds are drawn as squares with provenance color
            sink_nodes.append(node_id)
            sink_colors.append(color)
            sink_sizes.append(size)
            sink_edge_colors.append(edge_color)
        else:
            regular_nodes.append(node_id)
            regular_colors.append(color)
            regular_sizes.append(size)
            regular_edge_colors.append(edge_color)

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color='#bdc3c7',
        width=1.5,
        arrows=True,
        arrowsize=15,
        arrowstyle='-|>',
        connectionstyle='arc3,rad=0.1',
    )

    # Draw regular nodes (circles)
    if regular_nodes:
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=regular_nodes,
            node_color=regular_colors,
            node_size=regular_sizes,
            edgecolors=regular_edge_colors,
            linewidths=2,
        )

    # Draw sink compound nodes (squares)
    if sink_nodes:
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            nodelist=sink_nodes,
            node_color=sink_colors,
            node_size=sink_sizes,
            edgecolors=sink_edge_colors,
            linewidths=2,
            node_shape='s',  # Square shape for sink compounds
        )

    # Draw labels
    if show_smiles:
        labels = {}
        for node_id in G.nodes:
            node_data = G.nodes[node_id]
            smiles = node_data.get('smiles_short', str(node_id))
            if show_stats:
                visits = node_data.get('visits', 0)
                avg_val = node_data.get('avg_value', 0)
                labels[node_id] = f"{smiles}\nv={visits}, r={avg_val:.2f}"
            else:
                labels[node_id] = smiles

        nx.draw_networkx_labels(
            G, pos, labels, ax=ax,
            font_size=7,
            font_color='black',
            font_weight='bold',
        )
    else:
        # Just show node IDs
        labels = {n: str(n) for n in G.nodes}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        mpatches.Patch(facecolor='#f39c12', edgecolor='#2c3e50', label='Target'),
        mpatches.Patch(facecolor='#3498db', edgecolor='#2c3e50', label='Enzymatic'),
        mpatches.Patch(facecolor='#9b59b6', edgecolor='#2c3e50', label='Synthetic'),
        mpatches.Patch(facecolor='#2ecc71', edgecolor='#27ae60', linewidth=2, label='PKS Match ✓'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#7f8c8d',
               markeredgecolor='#2c3e50', markersize=12, label='■ = Sink (Building Block)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    # Title
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        target_smiles = Chem.MolToSmiles(agent.target_molecule) if agent.target_molecule else "Unknown"
        pks_matches = len([n for n in G.nodes if G.nodes[n].get('is_pks_match', False) and not G.nodes[n].get('is_sink_compound', False)])
        sink_compounds = len([n for n in G.nodes if G.nodes[n].get('is_sink_compound', False)])
        ax.set_title(
            f"DORAnet MCTS Search Tree\nTarget: {target_smiles}\n"
            f"Total Nodes: {len(G.nodes)}, PKS Matches: {pks_matches}, Sink Compounds: {sink_compounds}",
            fontsize=12, fontweight='bold'
        )

    # Add stats box
    sink_count = len(agent.get_sink_compounds()) if hasattr(agent, 'get_sink_compounds') else 0
    stats_text = (
        f"MCTS Statistics:\n"
        f"  Iterations: {agent.total_iterations}\n"
        f"  Max Depth: {agent.max_depth}\n"
        f"  Total Nodes: {len(agent.nodes)}\n"
        f"  PKS Library Size: {len(agent.pks_library)}\n"
        f"  Sink Compounds: {sink_count}\n"
        f"  RetroTide Spawned: {len(agent.retrotide_results)}"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    ax.set_axis_off()
    plt.tight_layout()

    # Save if path provided
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"[Visualization] Saved to: {path}")

    return fig


def visualize_pks_pathways(
    agent: "DORAnetMCTS",
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
) -> plt.Figure:
    """
    Create a focused visualization showing only pathways to PKS matches.

    Args:
        agent: The DORAnetMCTS agent after running.
        output_path: Path to save the figure (optional).
        figsize: Figure size in inches.

    Returns:
        Matplotlib Figure object.
    """
    from rdkit import Chem

    # Get PKS matching nodes
    pks_nodes = [n for n in agent.nodes if agent.calculate_reward(n) > 0]

    if not pks_nodes:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No PKS matches found", ha='center', va='center', fontsize=14)
        return fig

    # Create subgraph with only paths to PKS matches
    G = nx.DiGraph()

    # Add all nodes in paths to PKS matches
    nodes_in_paths = set()
    for pks_node in pks_nodes:
        current = pks_node
        while current is not None:
            nodes_in_paths.add(current.node_id)
            current = current.parent

    # Add nodes
    for node in agent.nodes:
        if node.node_id in nodes_in_paths:
            is_pks_match = agent._is_in_pks_library(node.smiles or "")
            G.add_node(
                node.node_id,
                smiles=node.smiles,
                depth=node.depth,
                provenance=node.provenance or "target",
                is_pks_match=is_pks_match,
            )

    # Add edges
    for parent_id, child_id in agent.edges:
        if parent_id in nodes_in_paths and child_id in nodes_in_paths:
            G.add_edge(parent_id, child_id)

    # Layout
    pos = get_hierarchical_pos(G, root=0)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Node colors
    node_colors = []
    for node_id in G.nodes:
        is_pks = G.nodes[node_id].get('is_pks_match', False)
        if is_pks:
            node_colors.append('#2ecc71')
        else:
            node_colors.append('#f39c12')

    # Draw
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#27ae60', width=3, arrows=True, arrowsize=20)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=1500, edgecolors='#27ae60', linewidths=3)

    # Labels with full SMILES
    labels = {n: G.nodes[n].get('smiles', str(n)) for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8, font_weight='bold')

    ax.set_title(f"Pathways to PKS-Synthesizable Fragments\n({len(pks_nodes)} matches found)",
                 fontsize=14, fontweight='bold')
    ax.set_axis_off()
    plt.tight_layout()

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"[Visualization] Saved to: {path}")

    return fig


def _generate_molecule_image_base64(smiles: str, size: Tuple[int, int] = (200, 200)) -> Optional[str]:
    """
    Generate a base64-encoded PNG image of a molecule from SMILES.

    Args:
        smiles: SMILES string of the molecule.
        size: Image size (width, height).

    Returns:
        Base64-encoded PNG image string, or None on failure.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        import base64
        from io import BytesIO

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Generate image
        img = Draw.MolToImage(mol, size=size)

        # Convert to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return f"data:image/png;base64,{img_str}"

    except Exception as e:
        print(f"[Visualization] Error generating molecule image: {e}")
        return None


def _generate_reaction_image_base64(
    reactants_smiles: List[str],
    products_smiles: List[str],
    size: Tuple[int, int] = (600, 200),
) -> Optional[str]:
    """
    Generate a base64-encoded PNG image of a chemical reaction.

    Args:
        reactants_smiles: List of reactant SMILES strings.
        products_smiles: List of product SMILES strings.
        size: Sub-image size per molecule (width, height).

    Returns:
        Base64-encoded PNG data URI, or None on failure.
    """
    try:
        from rdkit.Chem import AllChem, Draw
        import base64
        from io import BytesIO

        if not reactants_smiles and not products_smiles:
            return None

        rxn_smiles = f"{'.'.join(reactants_smiles)}>>{'.'.join(products_smiles)}"
        rxn = AllChem.ReactionFromSmarts(rxn_smiles, useSmiles=True)
        if rxn is None:
            return None

        img = Draw.ReactionToImage(rxn, subImgSize=size)

        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

    except Exception as e:
        print(f"[Visualization] Error generating reaction image: {e}")
        return None


def _compute_ucb1_scores(agent: "DORAnetMCTS") -> Dict[int, str]:
    """
    Compute the current UCB1 selection score for every node in the tree.

    Replicates the formula used in DORAnetMCTS.select() so the hover tooltip
    shows the score that will drive selection in the next iteration.

    Returns:
        Mapping from node_id -> formatted UCB1 string (or "N/A" for root).
    """
    scores: Dict[int, str] = {}
    policy = getattr(agent, "selection_policy", "depth_biased")
    depth_coeff = getattr(agent, "depth_bonus_coefficient", 0.0)

    for node in agent.nodes:
        if node.parent is None:
            scores[node.node_id] = "N/A"
            continue

        parent_visits = max(node.parent.visits, 1)

        if node.visits == 0:
            if policy == "depth_biased":
                raw = 1000.0 + depth_coeff * node.depth
            else:
                raw = float("inf")
        else:
            exploit = node.value / node.visits
            explore = math.sqrt(2 * math.log(parent_visits) / node.visits)
            raw = exploit + explore
            if policy == "depth_biased":
                raw += depth_coeff * node.depth

        scores[node.node_id] = f"{raw:.3f}" if math.isfinite(raw) else "∞"

    return scores


def create_interactive_html(
    agent: "DORAnetMCTS",
    output_path: str,
) -> None:
    """
    Create an interactive HTML visualization using Bokeh.

    Args:
        agent: The DORAnetMCTS agent after running.
        output_path: Path to save the HTML file.
    """
    try:
        from bokeh.plotting import figure, save, output_file
        from bokeh.models import (
            GraphRenderer, Circle, MultiLine, StaticLayoutProvider,
            HoverTool, ColumnDataSource, LabelSet
        )
        from bokeh.palettes import Spectral4
        from bokeh.io import export_png
    except ImportError:
        print("[Visualization] Bokeh not installed. Install with: pip install bokeh")
        return

    from rdkit import Chem

    # Create graph
    G = create_tree_graph(agent)
    pos = get_hierarchical_pos(G, root=0)

    # Prepare data
    node_ids = list(G.nodes)
    x_coords = [pos[n][0] for n in node_ids]
    y_coords = [pos[n][1] for n in node_ids]

    colors = []
    sizes = []
    smiles_list = []
    visits_list = []
    values_list = []
    provenance_list = []
    pks_match_list = []
    sink_compound_list = []
    sink_type_list = []

    for n in node_ids:
        data = G.nodes[n]
        is_pks = data.get('is_pks_match', False)
        is_sink = data.get('is_sink_compound', False)
        prov = data.get('provenance', 'target')

        # Color based on provenance (same for sink and non-sink)
        if is_pks and not is_sink:
            colors.append('#2ecc71')  # Green for PKS match (non-sink)
        elif prov == 'enzymatic':
            colors.append('#3498db')  # Blue for enzymatic
        elif prov == 'synthetic':
            colors.append('#9b59b6')  # Purple for synthetic
        else:
            colors.append('#f39c12')  # Orange for target

        visits = data.get('visits', 1)
        sizes.append(10 + visits * 3)

        smiles_list.append(data.get('smiles', 'N/A'))
        visits_list.append(data.get('visits', 0))
        values_list.append(f"{data.get('avg_value', 0):.3f}")
        provenance_list.append(prov)
        pks_match_list.append('Yes ✓' if is_pks else 'No')
        sink_compound_list.append('■ Yes' if is_sink else 'No')
        sink_type_list.append(data.get('sink_compound_type', None) or 'N/A')

    # Create Bokeh plot
    output_file(output_path)

    p = figure(
        title=f"DORAnet MCTS Search Tree - {Chem.MolToSmiles(agent.target_molecule)}",
        width=1200, height=800,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        x_range=(-3, 3), y_range=(-3, 1),
    )

    # Data source
    source = ColumnDataSource(data=dict(
        x=x_coords,
        y=y_coords,
        node_id=node_ids,
        color=colors,
        size=sizes,
        smiles=smiles_list,
        visits=visits_list,
        avg_value=values_list,
        provenance=provenance_list,
        pks_match=pks_match_list,
        sink_compound=sink_compound_list,
        sink_compound_type=sink_type_list,
    ))

    # Draw edges
    edge_x = []
    edge_y = []
    for start, end in G.edges:
        edge_x.append([pos[start][0], pos[end][0]])
        edge_y.append([pos[start][1], pos[end][1]])

    p.multi_line(edge_x, edge_y, line_color='#bdc3c7', line_width=2)

    # Draw nodes
    p.scatter('x', 'y', source=source, size='size', color='color',
              line_color='#2c3e50', line_width=2, marker='circle')

    # Hover tool
    hover = HoverTool(tooltips=[
        ("Node ID", "@node_id"),
        ("SMILES", "@smiles"),
        ("Provenance", "@provenance"),
        ("Visits", "@visits"),
        ("Avg Value", "@avg_value"),
        ("PKS Match", "@pks_match"),
        ("Sink Compound", "@sink_compound"),
        ("Sink Type", "@sink_compound_type"),
    ])
    p.add_tools(hover)

    # Add labels
    labels = LabelSet(x='x', y='y', text='node_id', source=source,
                      text_font_size='10pt', text_align='center')
    p.add_layout(labels)

    p.axis.visible = False
    p.grid.visible = False

    save(p)
    print(f"[Visualization] Interactive HTML saved to: {output_path}")


def create_enhanced_interactive_html(
    agent: "DORAnetMCTS",
    output_path: str,
    molecule_img_size: Tuple[int, int] = (250, 250),
    auto_open: bool = False,
    title: Optional[str] = None,
) -> None:
    """
    Create an enhanced interactive HTML visualization with molecule images and reaction info.

    Features:
    - Hover over nodes to see molecule structure images
    - Node metadata (enzymatic/synthetic, PKS match, visits, value, depth)
    - Hover over edges to see reaction information
    - Same color scheme as static visualizations
    - Interactive zoom and pan

    Args:
        agent: The DORAnetMCTS agent after running.
        output_path: Path to save the HTML file.
        molecule_img_size: Size of molecule images in pixels (width, height).
        auto_open: If True, automatically open the HTML file in the default browser.
        title: Custom title for the plot. If None, uses default title.
    """
    try:
        from bokeh.plotting import figure, save, output_file
        from bokeh.models import HoverTool, ColumnDataSource, LabelSet, Legend, LegendItem
        from bokeh.layouts import column
        from bokeh.models.annotations import Title
    except ImportError:
        print("[Visualization] Bokeh not installed. Install with: pip install bokeh")
        return

    from rdkit import Chem

    # Fixed node size for all nodes (no longer varies by visits)
    FIXED_NODE_SIZE = 20

    print("[Visualization] Generating enhanced interactive visualization...")
    print("[Visualization] Creating molecule structure images...")

    # Create graph
    G = create_tree_graph(agent)
    if len(G.nodes) == 0:
        print("[Visualization] No nodes to visualize.")
        return

    pos = get_hierarchical_pos(G, root=0)

    # Compute UCB1 scores for all nodes
    ucb1_scores = _compute_ucb1_scores(agent)

    # Prepare node data with molecule images
    node_ids = []
    x_coords = []
    y_coords = []
    colors = []
    sizes = []
    markers = []  # For different shapes (circle vs square)
    smiles_list = []
    smiles_short_list = []
    visits_list = []
    values_list = []
    avg_values_list = []
    provenance_list = []
    pks_match_list = []
    sink_compound_list = []
    sink_type_list = []
    depth_list = []
    mol_images = []
    ucb1_list = []

    for n in G.nodes:
        data = G.nodes[n]
        is_pks = data.get('is_pks_match', False)
        is_sink = data.get('is_sink_compound', False)
        prov = data.get('provenance', 'target')
        smiles = data.get('smiles', '')
        smiles_short = data.get('smiles_short', str(n))

        # Color and shape based on node type (matches manuscript legend)
        depth = data.get('depth', 0)
        sink_type = data.get('sink_compound_type', None)
        is_root = (depth == 0 and prov == 'target')
        if is_root:
            color = '#FFD700'  # Gold star for root/target molecule
            marker = 'star'
        elif is_sink and sink_type == 'chemical':
            color = '#4472C4'  # Blue square for chemical building blocks
            marker = 'square'
        elif is_sink and sink_type == 'biological':
            color = '#548235'  # Green square for biological building blocks
            marker = 'square'
        elif is_pks:
            color = '#C55A11'  # Orange square for polyketides (PKS database match)
            marker = 'square'
        else:
            color = '#D9D9D9'  # Grey circle for general intermediate compounds
            marker = 'circle'

        visits = data.get('visits', 0)
        value = data.get('value', 0.0)
        avg_value = data.get('avg_value', 0.0)

        # Generate molecule image
        mol_img = _generate_molecule_image_base64(smiles, size=molecule_img_size)

        node_ids.append(n)
        x_coords.append(pos[n][0])
        y_coords.append(pos[n][1])
        colors.append(color)
        sizes.append(FIXED_NODE_SIZE)  # Fixed size for all nodes
        markers.append(marker)
        smiles_list.append(smiles if smiles else 'N/A')
        smiles_short_list.append(smiles_short)
        visits_list.append(visits)
        values_list.append(f"{value:.2f}")
        avg_values_list.append(f"{avg_value:.3f}")
        provenance_list.append(prov.capitalize())
        pks_match_list.append('✓ Yes' if is_pks else '✗ No')
        sink_compound_list.append('■ Yes' if is_sink else '✗ No')
        sink_type_list.append(data.get('sink_compound_type', None) or 'N/A')
        depth_list.append(depth)
        mol_images.append(mol_img if mol_img else "")
        ucb1_list.append(ucb1_scores.get(n, "N/A"))

    # Create node data source
    node_source = ColumnDataSource(data=dict(
        x=x_coords,
        y=y_coords,
        node_id=node_ids,
        color=colors,
        size=sizes,
        marker=markers,
        smiles=smiles_list,
        smiles_short=smiles_short_list,
        visits=visits_list,
        value=values_list,
        avg_value=avg_values_list,
        provenance=provenance_list,
        pks_match=pks_match_list,
        sink_compound=sink_compound_list,
        sink_compound_type=sink_type_list,
        depth=depth_list,
        mol_img=mol_images,
        ucb1=ucb1_list,
    ))

    # Prepare edge data with reaction information
    print("[Visualization] Extracting reaction information for edges...")
    edge_x0 = []
    edge_y0 = []
    edge_x1 = []
    edge_y1 = []
    edge_reactions = []
    edge_reaction_equations = []
    edge_colors = []
    edge_dora_xgb = []
    edge_delta_h = []
    edge_thermo_scaled = []
    edge_rxn_imgs = []

    for parent_id, child_id in agent.edges:
        if parent_id in pos and child_id in pos:
            # Get child node for reaction info
            child_node = next((n for n in agent.nodes if n.node_id == child_id), None)

            # Extract reaction information
            if child_node:
                rxn_label = child_node.reaction_name or "Unknown reaction"
                reactants = child_node.reactants_smiles or []
                products = child_node.products_smiles or []
                if reactants or products:
                    rxn_equation = f"{'.'.join(reactants)}>>{'.'.join(products)}"
                else:
                    rxn_equation = "N/A"

                # Truncate for tooltip
                rxn_label_short = rxn_label[:100] + "..." if len(rxn_label) > 100 else rxn_label
                rxn_equation_short = rxn_equation[:140] + "..." if len(rxn_equation) > 140 else rxn_equation

                reaction_info = rxn_label_short
                reaction_equation_info = rxn_equation_short

                # Edge color based on reaction type (matches manuscript legend)
                if child_node.provenance == 'enzymatic':
                    edge_color = '#548235'  # Green for enzymatic reactions
                elif child_node.provenance == 'synthetic':
                    edge_color = '#4472C4'  # Blue for chemical/synthetic reactions
                else:
                    edge_color = '#95a5a6'  # Grey for unknown

                # Thermodynamic/feasibility scores for edge hover tooltip
                if child_node.provenance == 'enzymatic':
                    dora_xgb_val = f"{child_node.feasibility_score:.3f}" if child_node.feasibility_score is not None else "N/A"
                    delta_h_val = "N/A"
                    thermo_scaled_val = "N/A"
                elif child_node.provenance == 'synthetic':
                    dora_xgb_val = "N/A"
                    if child_node.enthalpy_of_reaction is not None:
                        dh = child_node.enthalpy_of_reaction
                        scaled = 1.0 / (1.0 + math.exp(0.2 * (dh - 15.0)))
                        delta_h_val = f"{dh:.2f} kcal/mol"
                        thermo_scaled_val = f"{scaled:.3f}"
                    else:
                        delta_h_val = "N/A"
                        thermo_scaled_val = "N/A"
                else:
                    dora_xgb_val = "N/A"
                    delta_h_val = "N/A"
                    thermo_scaled_val = "N/A"

                rxn_img = _generate_reaction_image_base64(
                    child_node.reactants_smiles or [],
                    child_node.products_smiles or [],
                )
                rxn_img_val = rxn_img if rxn_img else ""
            else:
                reaction_info = "No reaction information"
                reaction_equation_info = "N/A"
                edge_color = '#95a5a6'
                dora_xgb_val = "N/A"
                delta_h_val = "N/A"
                thermo_scaled_val = "N/A"
                rxn_img_val = ""

            edge_x0.append(pos[parent_id][0])
            edge_y0.append(pos[parent_id][1])
            edge_x1.append(pos[child_id][0])
            edge_y1.append(pos[child_id][1])
            edge_reactions.append(reaction_info)
            edge_reaction_equations.append(reaction_equation_info)
            edge_colors.append(edge_color)
            edge_dora_xgb.append(dora_xgb_val)
            edge_delta_h.append(delta_h_val)
            edge_thermo_scaled.append(thermo_scaled_val)
            edge_rxn_imgs.append(rxn_img_val)

    # Create edge data source
    edge_source = ColumnDataSource(data=dict(
        x0=edge_x0,
        y0=edge_y0,
        x1=edge_x1,
        y1=edge_y1,
        reaction=edge_reactions,
        reaction_equation=edge_reaction_equations,
        color=edge_colors,
        dora_xgb=edge_dora_xgb,
        delta_h=edge_delta_h,
        thermo_scaled=edge_thermo_scaled,
        rxn_img=edge_rxn_imgs,
    ))

    # Create Bokeh figure
    target_smiles = Chem.MolToSmiles(agent.target_molecule) if agent.target_molecule else "Unknown"
    pks_matches = len([n for n in G.nodes if G.nodes[n].get('is_pks_match', False) and not G.nodes[n].get('is_sink_compound', False)])
    sink_compounds = len([n for n in G.nodes if G.nodes[n].get('is_sink_compound', False)])

    # Use custom title if provided, otherwise use default
    plot_title = title if title else "DORAnet MCTS Interactive Search Tree"

    # Compute explicit data ranges to prevent off-screen legend glyphs from affecting view
    x_min = min(x_coords) if x_coords else -1
    x_max = max(x_coords) if x_coords else 1
    y_min = min(y_coords) if y_coords else -1
    y_max = max(y_coords) if y_coords else 1
    # Add padding (10% on each side)
    x_padding = (x_max - x_min) * 0.1 if x_max != x_min else 1
    y_padding = (y_max - y_min) * 0.1 if y_max != y_min else 1

    from bokeh.models import Range1d
    p = figure(
        title=plot_title,
        width=1400,
        height=900,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
        x_range=Range1d(x_min - x_padding, x_max + x_padding),
        y_range=Range1d(y_min - y_padding, y_max + y_padding),
    )

    # Add subtitle with target info
    p.add_layout(Title(
        text=f"Target: {target_smiles[:100]} | Total Nodes: {len(G.nodes)} | PKS Matches: {pks_matches} | Sink Compounds: {sink_compounds}",
        text_font_size="11pt",
        text_font_style="italic"
    ), 'above')

    # Draw edges with hover
    edge_lines = p.segment(
        x0='x0', y0='y0', x1='x1', y1='y1',
        source=edge_source,
        color='color',
        line_width=2.5,
        alpha=0.7,
        line_cap='round'
    )

    # Edge hover tool with HTML template for reaction images
    edge_hover_html = """
<div style="width: 360px; background-color: white; border: 2px solid #555; border-radius: 8px; padding: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
    <div style="text-align: center; margin-bottom: 8px;">
        <img src="@rxn_img" style="max-width: 340px; max-height: 160px; border: 1px solid #ddd; border-radius: 4px;">
    </div>
    <div style="font-family: monospace; font-size: 12px;">
        <b style="color: #2c3e50;">Reaction:</b> @reaction<br>
        <b style="color: #2c3e50;">Equation:</b> <span style="font-size: 10px; word-break: break-all;">@reaction_equation</span><br>
        <b style="color: #2c3e50;">DORA-XGB score:</b> @dora_xgb<br>
        <b style="color: #2c3e50;">ΔH (PAthermo):</b> @delta_h<br>
        <b style="color: #2c3e50;">PAthermo scaled:</b> @thermo_scaled<br>
    </div>
</div>
"""
    edge_hover = HoverTool(
        renderers=[edge_lines],
        tooltips=edge_hover_html,
        point_policy="follow_mouse"
    )
    p.add_tools(edge_hover)

    # Draw nodes split by marker type
    circle_indices = [i for i, m in enumerate(markers) if m == 'circle']
    square_indices = [i for i, m in enumerate(markers) if m == 'square']
    star_indices   = [i for i, m in enumerate(markers) if m == 'star']

    renderers_to_hover = []

    if circle_indices:
        circle_source = ColumnDataSource(data={k: [v[i] for i in circle_indices] for k, v in node_source.data.items()})
        node_circles = p.scatter('x', 'y', source=circle_source, size='size', fill_color='color',
                                 line_color='#2c3e50', line_width=2.5, alpha=0.9, marker='circle')
        renderers_to_hover.append(node_circles)

    if square_indices:
        square_source = ColumnDataSource(data={k: [v[i] for i in square_indices] for k, v in node_source.data.items()})
        node_squares = p.scatter('x', 'y', source=square_source, size='size', fill_color='color',
                                 line_color='#2c3e50', line_width=2.5, alpha=0.9, marker='square')
        renderers_to_hover.append(node_squares)

    if star_indices:
        star_source = ColumnDataSource(data={k: [v[i] for i in star_indices] for k, v in node_source.data.items()})
        node_stars = p.scatter('x', 'y', source=star_source, size='size', fill_color='color',
                               line_color='#2c3e50', line_width=2.5, alpha=0.9, marker='star')
        renderers_to_hover.append(node_stars)

    # Node hover tool with custom HTML template for molecule images
    node_hover_html = """
    <div style="width: 320px; background-color: white; border: 2px solid #2c3e50; border-radius: 8px; padding: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <div style="text-align: center; margin-bottom: 10px;">
            <img src="@mol_img" style="max-width: 250px; max-height: 250px; border: 1px solid #ddd; border-radius: 4px;">
        </div>
        <div style="font-family: monospace; font-size: 12px;">
            <b style="color: #2c3e50;">Node ID:</b> @node_id<br>
            <b style="color: #2c3e50;">Depth:</b> @depth<br>
            <b style="color: #2c3e50;">Provenance:</b> <span style="color: @color; font-weight: bold;">@provenance</span><br>
            <b style="color: #2c3e50;">PKS Match:</b> @pks_match<br>
            <b style="color: #00bcd4;">Sink Compound:</b> @sink_compound<br>
            <b style="color: #00bcd4;">Sink Type:</b> @sink_compound_type<br>
            <b style="color: #2c3e50;">Visits:</b> @visits<br>
            <b style="color: #2c3e50;">Avg Value:</b> @avg_value<br>
            <b style="color: #2c3e50;">UCB1 score:</b> @ucb1<br>
            <b style="color: #2c3e50;">SMILES:</b><br>
            <span style="font-size: 10px; word-break: break-all;">@smiles</span>
        </div>
    </div>
    """

    if renderers_to_hover:
        node_hover = HoverTool(renderers=renderers_to_hover, tooltips=node_hover_html, point_policy="follow_mouse")
        p.add_tools(node_hover)

    # Add node ID labels
    labels = LabelSet(
        x='x', y='y',
        text='node_id',
        source=node_source,
        text_font_size='9pt',
        text_align='center',
        text_baseline='middle',
        text_color='black'
        # Note: text_font_weight not supported in LabelSet
    )
    p.add_layout(labels)

    # Create legend with dummy glyphs for node types and edge types
    # Use off-screen coordinates so the glyphs don't appear in the plot but show in legend
    # We place them far outside the visible plot range
    OFF_SCREEN = -99999

    legend_items = []

    # Target molecule (gold star)
    target_glyph = p.scatter([OFF_SCREEN], [OFF_SCREEN], marker='star', size=15, fill_color='#FFD700',
                             line_color='#2c3e50', line_width=2)
    legend_items.append(LegendItem(label="Target Molecule", renderers=[target_glyph]))

    # Chemical building block (blue square)
    chem_glyph = p.scatter([OFF_SCREEN], [OFF_SCREEN], marker='square', size=15, fill_color='#4472C4',
                           line_color='#2c3e50', line_width=2)
    legend_items.append(LegendItem(label="Chemical Building Block", renderers=[chem_glyph]))

    # Biological building block (green square)
    bio_glyph = p.scatter([OFF_SCREEN], [OFF_SCREEN], marker='square', size=15, fill_color='#548235',
                          line_color='#2c3e50', line_width=2)
    legend_items.append(LegendItem(label="Biological Building Block", renderers=[bio_glyph]))

    # PKS database match (orange square)
    pks_glyph = p.scatter([OFF_SCREEN], [OFF_SCREEN], marker='square', size=15, fill_color='#C55A11',
                          line_color='#2c3e50', line_width=2)
    legend_items.append(LegendItem(label="PKS Database Match", renderers=[pks_glyph]))

    # Intermediate compound (grey circle)
    int_glyph = p.scatter([OFF_SCREEN], [OFF_SCREEN], marker='circle', size=15, fill_color='#D9D9D9',
                          line_color='#2c3e50', line_width=2)
    legend_items.append(LegendItem(label="Intermediate Compound", renderers=[int_glyph]))

    # Edge type legend items (enzymatic = green, synthetic = blue)
    enz_edge = p.line([OFF_SCREEN, OFF_SCREEN], [OFF_SCREEN, OFF_SCREEN], line_color='#548235', line_width=3)
    legend_items.append(LegendItem(label="Enzymatic Reaction", renderers=[enz_edge]))

    syn_edge = p.line([OFF_SCREEN, OFF_SCREEN], [OFF_SCREEN, OFF_SCREEN], line_color='#4472C4', line_width=3)
    legend_items.append(LegendItem(label="Synthetic Reaction", renderers=[syn_edge]))

    # Add legend to plot
    legend = Legend(items=legend_items, location="top_left", title="Legend",
                    title_text_font_style="bold", label_text_font_size="10pt",
                    background_fill_alpha=0.8, border_line_color="#2c3e50")
    p.add_layout(legend, 'right')

    # Style the plot
    p.axis.visible = False
    p.grid.visible = False
    p.background_fill_color = "#f8f9fa"
    p.border_fill_color = "#ffffff"

    # Save to HTML
    output_file(output_path)
    save(p)

    print(f"[Visualization] Enhanced interactive HTML saved to: {output_path}")
    print(f"[Visualization] Open in browser to explore:")
    print(f"[Visualization]   - Hover over nodes to see molecule structures")
    print(f"[Visualization]   - Hover over edges to see reactions")
    print(f"[Visualization]   - Use mouse wheel to zoom, drag to pan")

    # Auto-open in browser if requested
    if auto_open:
        import webbrowser
        from pathlib import Path

        # Convert to absolute path and open
        abs_path = Path(output_path).resolve()
        webbrowser.open(f"file://{abs_path}")
        print(f"[Visualization] Opening visualization in your default browser...")


def create_pathways_interactive_html(
    agent: "DORAnetMCTS",
    output_path: str,
    molecule_img_size: Tuple[int, int] = (250, 250),
    auto_open: bool = False,
    title: Optional[str] = None,
) -> None:
    """
    Create an interactive HTML visualization showing ONLY pathways to PKS matches and sink compounds.

    This filtered view helps focus on successful retrosynthetic routes by excluding
    nodes that did not lead to any valuable building blocks.

    Features:
    - Shows only nodes that are PKS matches, sink compounds, or ancestors of these
    - Same hover functionality as the full tree visualization
    - Cleaner view of successful pathways

    Args:
        agent: The DORAnetMCTS agent after running.
        output_path: Path to save the HTML file.
        molecule_img_size: Size of molecule images in pixels (width, height).
        auto_open: If True, automatically open the HTML file in the default browser.
        title: Custom title for the plot. If None, uses default title.
    """
    try:
        from bokeh.plotting import figure, save, output_file
        from bokeh.models import HoverTool, ColumnDataSource, LabelSet, Legend, LegendItem
        from bokeh.layouts import column
        from bokeh.models.annotations import Title
    except ImportError:
        print("[Visualization] Bokeh not installed. Install with: pip install bokeh")
        return

    from rdkit import Chem

    # Fixed node size for all nodes (no longer varies by visits)
    FIXED_NODE_SIZE = 20

    print("[Visualization] Generating pathways-only interactive visualization...")

    # First, identify all terminal nodes (PKS matches and sink compounds)
    terminal_nodes = set()
    for node in agent.nodes:
        is_pks = agent._is_in_pks_library(node.smiles or "")
        is_sink = node.is_sink_compound
        if is_pks or is_sink:
            terminal_nodes.add(node.node_id)

    if not terminal_nodes:
        print("[Visualization] No PKS matches or sink compounds found. Skipping pathways visualization.")
        return

    # Trace back from each terminal node to find all nodes in pathways
    nodes_in_pathways = set()
    for node in agent.nodes:
        if node.node_id in terminal_nodes:
            # Trace back to root
            current = node
            while current is not None:
                nodes_in_pathways.add(current.node_id)
                current = current.parent

    print(f"[Visualization] Found {len(terminal_nodes)} terminal nodes (PKS/sink), "
          f"{len(nodes_in_pathways)} nodes in pathways")

    # Create filtered graph
    G = create_tree_graph(agent)

    # Filter to only include nodes in pathways
    nodes_to_keep = nodes_in_pathways
    G_filtered = G.subgraph(nodes_to_keep).copy()

    if len(G_filtered.nodes) == 0:
        print("[Visualization] No nodes to visualize after filtering.")
        return

    pos = get_hierarchical_pos(G_filtered, root=0)

    # Compute UCB1 scores for all nodes
    ucb1_scores = _compute_ucb1_scores(agent)

    # Prepare node data with molecule images
    print("[Visualization] Creating molecule structure images for pathways...")
    node_ids = []
    x_coords = []
    y_coords = []
    colors = []
    sizes = []
    markers = []
    smiles_list = []
    smiles_short_list = []
    visits_list = []
    values_list = []
    avg_values_list = []
    provenance_list = []
    pks_match_list = []
    sink_compound_list = []
    sink_type_list = []
    depth_list = []
    mol_images = []
    ucb1_list = []

    for n in G_filtered.nodes:
        data = G_filtered.nodes[n]
        is_pks = data.get('is_pks_match', False)
        is_sink = data.get('is_sink_compound', False)
        prov = data.get('provenance', 'target')
        smiles = data.get('smiles', '')
        smiles_short = data.get('smiles_short', str(n))

        # Color and shape based on node type (matches manuscript legend)
        depth = data.get('depth', 0)
        sink_type = data.get('sink_compound_type', None)
        is_root = (depth == 0 and prov == 'target')
        if is_root:
            color = '#FFD700'  # Gold star for root/target molecule
            marker = 'star'
        elif is_sink and sink_type == 'chemical':
            color = '#4472C4'  # Blue square for chemical building blocks
            marker = 'square'
        elif is_sink and sink_type == 'biological':
            color = '#548235'  # Green square for biological building blocks
            marker = 'square'
        elif is_pks:
            color = '#C55A11'  # Orange square for polyketides (PKS database match)
            marker = 'square'
        else:
            color = '#D9D9D9'  # Grey circle for general intermediate compounds
            marker = 'circle'

        visits = data.get('visits', 0)
        value = data.get('value', 0.0)
        avg_value = data.get('avg_value', 0.0)

        # Generate molecule image
        mol_img = _generate_molecule_image_base64(smiles, size=molecule_img_size)

        node_ids.append(n)
        x_coords.append(pos[n][0])
        y_coords.append(pos[n][1])
        colors.append(color)
        sizes.append(FIXED_NODE_SIZE)  # Fixed size for all nodes
        markers.append(marker)
        smiles_list.append(smiles if smiles else 'N/A')
        smiles_short_list.append(smiles_short)
        visits_list.append(visits)
        values_list.append(f"{value:.2f}")
        avg_values_list.append(f"{avg_value:.3f}")
        provenance_list.append(prov.capitalize())
        pks_match_list.append('✓ Yes' if is_pks else '✗ No')
        sink_compound_list.append('■ Yes' if is_sink else '✗ No')
        sink_type_list.append(data.get('sink_compound_type', None) or 'N/A')
        depth_list.append(depth)
        mol_images.append(mol_img if mol_img else "")
        ucb1_list.append(ucb1_scores.get(n, "N/A"))

    # Create node data source
    node_source = ColumnDataSource(data=dict(
        x=x_coords,
        y=y_coords,
        node_id=node_ids,
        color=colors,
        size=sizes,
        marker=markers,
        smiles=smiles_list,
        smiles_short=smiles_short_list,
        visits=visits_list,
        value=values_list,
        avg_value=avg_values_list,
        provenance=provenance_list,
        pks_match=pks_match_list,
        sink_compound=sink_compound_list,
        sink_compound_type=sink_type_list,
        depth=depth_list,
        mol_img=mol_images,
        ucb1=ucb1_list,
    ))

    # Prepare edge data
    edge_x0 = []
    edge_y0 = []
    edge_x1 = []
    edge_y1 = []
    edge_reactions = []
    edge_reaction_equations = []
    edge_colors = []
    edge_dora_xgb = []
    edge_delta_h = []
    edge_thermo_scaled = []
    edge_rxn_imgs = []

    for parent_id, child_id in agent.edges:
        # Only include edges where both nodes are in the filtered set
        if parent_id in nodes_to_keep and child_id in nodes_to_keep:
            if parent_id in pos and child_id in pos:
                child_node = next((n for n in agent.nodes if n.node_id == child_id), None)

                if child_node:
                    rxn_label = child_node.reaction_name or "Unknown reaction"
                    reactants = child_node.reactants_smiles or []
                    products = child_node.products_smiles or []
                    if reactants or products:
                        rxn_equation = f"{'.'.join(reactants)}>>{'.'.join(products)}"
                    else:
                        rxn_equation = "N/A"
                    rxn_label_short = rxn_label[:100] + "..." if len(rxn_label) > 100 else rxn_label
                    rxn_equation_short = rxn_equation[:140] + "..." if len(rxn_equation) > 140 else rxn_equation
                    reaction_info = rxn_label_short
                    reaction_equation_info = rxn_equation_short

                    # Edge color based on reaction type (matches manuscript legend)
                    if child_node.provenance == 'enzymatic':
                        edge_color = '#548235'  # Green for enzymatic reactions
                    elif child_node.provenance == 'synthetic':
                        edge_color = '#4472C4'  # Blue for chemical/synthetic reactions
                    else:
                        edge_color = '#95a5a6'  # Grey for unknown

                    # Thermodynamic/feasibility scores for edge hover tooltip
                    if child_node.provenance == 'enzymatic':
                        dora_xgb_val = f"{child_node.feasibility_score:.3f}" if child_node.feasibility_score is not None else "N/A"
                        delta_h_val = "N/A"
                        thermo_scaled_val = "N/A"
                    elif child_node.provenance == 'synthetic':
                        dora_xgb_val = "N/A"
                        if child_node.enthalpy_of_reaction is not None:
                            dh = child_node.enthalpy_of_reaction
                            scaled = 1.0 / (1.0 + math.exp(0.2 * (dh - 15.0)))
                            delta_h_val = f"{dh:.2f} kcal/mol"
                            thermo_scaled_val = f"{scaled:.3f}"
                        else:
                            delta_h_val = "N/A"
                            thermo_scaled_val = "N/A"
                    else:
                        dora_xgb_val = "N/A"
                        delta_h_val = "N/A"
                        thermo_scaled_val = "N/A"

                    rxn_img = _generate_reaction_image_base64(
                        child_node.reactants_smiles or [],
                        child_node.products_smiles or [],
                    )
                    rxn_img_val = rxn_img if rxn_img else ""
                else:
                    reaction_info = "No reaction information"
                    reaction_equation_info = "N/A"
                    edge_color = '#95a5a6'
                    dora_xgb_val = "N/A"
                    delta_h_val = "N/A"
                    thermo_scaled_val = "N/A"
                    rxn_img_val = ""

                edge_x0.append(pos[parent_id][0])
                edge_y0.append(pos[parent_id][1])
                edge_x1.append(pos[child_id][0])
                edge_y1.append(pos[child_id][1])
                edge_reactions.append(reaction_info)
                edge_reaction_equations.append(reaction_equation_info)
                edge_colors.append(edge_color)
                edge_dora_xgb.append(dora_xgb_val)
                edge_delta_h.append(delta_h_val)
                edge_thermo_scaled.append(thermo_scaled_val)
                edge_rxn_imgs.append(rxn_img_val)

    edge_source = ColumnDataSource(data=dict(
        x0=edge_x0,
        y0=edge_y0,
        x1=edge_x1,
        y1=edge_y1,
        reaction=edge_reactions,
        reaction_equation=edge_reaction_equations,
        color=edge_colors,
        dora_xgb=edge_dora_xgb,
        delta_h=edge_delta_h,
        thermo_scaled=edge_thermo_scaled,
        rxn_img=edge_rxn_imgs,
    ))

    # Create Bokeh figure
    target_smiles = Chem.MolToSmiles(agent.target_molecule) if agent.target_molecule else "Unknown"
    pks_matches = len([n for n in G_filtered.nodes if G_filtered.nodes[n].get('is_pks_match', False) and not G_filtered.nodes[n].get('is_sink_compound', False)])
    sink_compounds = len([n for n in G_filtered.nodes if G_filtered.nodes[n].get('is_sink_compound', False)])

    plot_title = title if title else "DORAnet MCTS - Pathways to PKS Matches & Building Blocks"

    # Compute explicit data ranges to prevent off-screen legend glyphs from affecting view
    x_min = min(x_coords) if x_coords else -1
    x_max = max(x_coords) if x_coords else 1
    y_min = min(y_coords) if y_coords else -1
    y_max = max(y_coords) if y_coords else 1
    # Add padding (10% on each side)
    x_padding = (x_max - x_min) * 0.1 if x_max != x_min else 1
    y_padding = (y_max - y_min) * 0.1 if y_max != y_min else 1

    from bokeh.models import Range1d
    p = figure(
        title=plot_title,
        width=1400,
        height=900,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
        x_range=Range1d(x_min - x_padding, x_max + x_padding),
        y_range=Range1d(y_min - y_padding, y_max + y_padding),
    )

    # Add subtitle
    p.add_layout(Title(
        text=f"Target: {target_smiles[:80]} | Pathway Nodes: {len(G_filtered.nodes)} | PKS: {pks_matches} | Sink: {sink_compounds}",
        text_font_size="11pt",
        text_font_style="italic"
    ), 'above')

    # Draw edges
    edge_lines = p.segment(
        x0='x0', y0='y0', x1='x1', y1='y1',
        source=edge_source,
        color='color',
        line_width=2.5,
        alpha=0.7,
    )

    edge_hover_html = """
<div style="width: 360px; background-color: white; border: 2px solid #555; border-radius: 8px; padding: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
    <div style="text-align: center; margin-bottom: 8px;">
        <img src="@rxn_img" style="max-width: 340px; max-height: 160px; border: 1px solid #ddd; border-radius: 4px;">
    </div>
    <div style="font-family: monospace; font-size: 12px;">
        <b style="color: #2c3e50;">Reaction:</b> @reaction<br>
        <b style="color: #2c3e50;">Equation:</b> <span style="font-size: 10px; word-break: break-all;">@reaction_equation</span><br>
        <b style="color: #2c3e50;">DORA-XGB score:</b> @dora_xgb<br>
        <b style="color: #2c3e50;">ΔH (PAthermo):</b> @delta_h<br>
        <b style="color: #2c3e50;">PAthermo scaled:</b> @thermo_scaled<br>
    </div>
</div>
"""
    edge_hover = HoverTool(
        renderers=[edge_lines],
        tooltips=edge_hover_html,
        line_policy="interp"
    )
    p.add_tools(edge_hover)

    # Separate nodes by marker type for drawing
    circle_indices = [i for i, m in enumerate(markers) if m == 'circle']
    square_indices = [i for i, m in enumerate(markers) if m == 'square']
    star_indices   = [i for i, m in enumerate(markers) if m == 'star']

    # Create separate data sources for circles, squares, and stars
    if circle_indices:
        circle_source = ColumnDataSource(data={
            key: [values[i] for i in circle_indices]
            for key, values in node_source.data.items()
        })
        node_circles = p.scatter(
            'x', 'y',
            source=circle_source,
            size='size',
            fill_color='color',
            line_color='#2c3e50',
            line_width=2,
            alpha=0.9,
            marker='circle'
        )

    if square_indices:
        square_source = ColumnDataSource(data={
            key: [values[i] for i in square_indices]
            for key, values in node_source.data.items()
        })
        node_squares = p.scatter(
            'x', 'y',
            source=square_source,
            size='size',
            fill_color='color',
            line_color='#2c3e50',
            line_width=2,
            alpha=0.9,
            marker='square'
        )

    if star_indices:
        star_source = ColumnDataSource(data={
            key: [values[i] for i in star_indices]
            for key, values in node_source.data.items()
        })
        node_stars = p.scatter(
            'x', 'y',
            source=star_source,
            size='size',
            fill_color='color',
            line_color='#2c3e50',
            line_width=2,
            alpha=0.9,
            marker='star'
        )

    # Node hover tool
    node_hover_html = """
    <div style="width: 320px; background-color: white; border: 2px solid #2c3e50; border-radius: 8px; padding: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <div style="text-align: center; margin-bottom: 10px;">
            <img src="@mol_img" style="max-width: 250px; max-height: 250px; border: 1px solid #ddd; border-radius: 4px;">
        </div>
        <div style="font-family: monospace; font-size: 12px;">
            <b style="color: #2c3e50;">Node ID:</b> @node_id<br>
            <b style="color: #2c3e50;">Depth:</b> @depth<br>
            <b style="color: #2c3e50;">Provenance:</b> <span style="color: @color; font-weight: bold;">@provenance</span><br>
            <b style="color: #2c3e50;">PKS Match:</b> @pks_match<br>
            <b style="color: #2c3e50;">Sink Compound:</b> @sink_compound<br>
            <b style="color: #2c3e50;">Sink Type:</b> @sink_compound_type<br>
            <b style="color: #2c3e50;">Visits:</b> @visits<br>
            <b style="color: #2c3e50;">Avg Value:</b> @avg_value<br>
            <b style="color: #2c3e50;">UCB1 score:</b> @ucb1<br>
            <b style="color: #2c3e50;">SMILES:</b><br>
            <span style="font-size: 10px; word-break: break-all;">@smiles</span>
        </div>
    </div>
    """

    renderers_to_hover = []
    if circle_indices:
        renderers_to_hover.append(node_circles)
    if square_indices:
        renderers_to_hover.append(node_squares)
    if star_indices:
        renderers_to_hover.append(node_stars)

    if renderers_to_hover:
        node_hover = HoverTool(
            renderers=renderers_to_hover,
            tooltips=node_hover_html,
            point_policy="follow_mouse"
        )
        p.add_tools(node_hover)

    # Add node ID labels
    labels = LabelSet(
        x='x', y='y',
        text='node_id',
        source=node_source,
        text_font_size='9pt',
        text_align='center',
        text_baseline='middle',
        text_color='black'
    )
    p.add_layout(labels)

    # Create legend with dummy glyphs for node types and edge types
    # Use off-screen coordinates so the glyphs don't appear in the plot but show in legend
    OFF_SCREEN = -99999

    legend_items = []

    # Target molecule (gold star)
    target_glyph = p.scatter([OFF_SCREEN], [OFF_SCREEN], marker='star', size=15, fill_color='#FFD700',
                             line_color='#2c3e50', line_width=2)
    legend_items.append(LegendItem(label="Target Molecule", renderers=[target_glyph]))

    # Chemical building block (blue square)
    chem_glyph = p.scatter([OFF_SCREEN], [OFF_SCREEN], marker='square', size=15, fill_color='#4472C4',
                           line_color='#2c3e50', line_width=2)
    legend_items.append(LegendItem(label="Chemical Building Block", renderers=[chem_glyph]))

    # Biological building block (green square)
    bio_glyph = p.scatter([OFF_SCREEN], [OFF_SCREEN], marker='square', size=15, fill_color='#548235',
                          line_color='#2c3e50', line_width=2)
    legend_items.append(LegendItem(label="Biological Building Block", renderers=[bio_glyph]))

    # PKS database match (orange square)
    pks_glyph = p.scatter([OFF_SCREEN], [OFF_SCREEN], marker='square', size=15, fill_color='#C55A11',
                          line_color='#2c3e50', line_width=2)
    legend_items.append(LegendItem(label="PKS Database Match", renderers=[pks_glyph]))

    # Intermediate compound (grey circle)
    int_glyph = p.scatter([OFF_SCREEN], [OFF_SCREEN], marker='circle', size=15, fill_color='#D9D9D9',
                          line_color='#2c3e50', line_width=2)
    legend_items.append(LegendItem(label="Intermediate Compound", renderers=[int_glyph]))

    # Edge type legend items (enzymatic = green, synthetic = blue)
    enz_edge = p.line([OFF_SCREEN, OFF_SCREEN], [OFF_SCREEN, OFF_SCREEN], line_color='#548235', line_width=3)
    legend_items.append(LegendItem(label="Enzymatic Reaction", renderers=[enz_edge]))

    syn_edge = p.line([OFF_SCREEN, OFF_SCREEN], [OFF_SCREEN, OFF_SCREEN], line_color='#4472C4', line_width=3)
    legend_items.append(LegendItem(label="Synthetic Reaction", renderers=[syn_edge]))

    # Add legend to plot
    legend = Legend(items=legend_items, location="top_left", title="Legend",
                    title_text_font_style="bold", label_text_font_size="10pt",
                    background_fill_alpha=0.8, border_line_color="#2c3e50")
    p.add_layout(legend, 'right')

    # Style the plot
    p.axis.visible = False
    p.grid.visible = False
    p.background_fill_color = "#f8f9fa"
    p.border_fill_color = "#ffffff"

    # Save to HTML
    output_file(output_path)
    save(p)

    print(f"[Visualization] Pathways interactive HTML saved to: {output_path}")

    # Auto-open in browser if requested
    if auto_open:
        import webbrowser
        from pathlib import Path

        abs_path = Path(output_path).resolve()
        webbrowser.open(f"file://{abs_path}")
        print(f"[Visualization] Opening pathways visualization in your default browser...")
