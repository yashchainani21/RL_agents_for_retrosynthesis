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
        is_pks_match = agent.calculate_reward(node) > 0
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

    # Prepare node colors based on PKS match and provenance
    node_colors = []
    node_sizes = []
    node_edge_colors = []

    for node_id in G.nodes:
        node_data = G.nodes[node_id]
        is_pks = node_data.get('is_pks_match', False)
        provenance = node_data.get('provenance', 'target')
        visits = node_data.get('visits', 1)

        # Color based on PKS match
        if is_pks:
            color = '#2ecc71'  # Green for PKS match
        elif provenance == 'enzymatic':
            color = '#3498db'  # Blue for enzymatic
        elif provenance == 'synthetic':
            color = '#9b59b6'  # Purple for synthetic
        else:
            color = '#f39c12'  # Orange for target

        node_colors.append(color)

        # Size based on visits (min 300, max 2000)
        size = min(300 + visits * 100, 2000)
        node_sizes.append(size)

        # Edge color: thick green border for PKS matches
        if is_pks:
            node_edge_colors.append('#27ae60')
        else:
            node_edge_colors.append('#2c3e50')

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

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors=node_edge_colors,
        linewidths=2,
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
    legend_elements = [
        mpatches.Patch(facecolor='#f39c12', edgecolor='#2c3e50', label='Target'),
        mpatches.Patch(facecolor='#3498db', edgecolor='#2c3e50', label='Enzymatic'),
        mpatches.Patch(facecolor='#9b59b6', edgecolor='#2c3e50', label='Synthetic'),
        mpatches.Patch(facecolor='#2ecc71', edgecolor='#27ae60', linewidth=2, label='PKS Match ✓'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    # Title
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        target_smiles = Chem.MolToSmiles(agent.target_molecule) if agent.target_molecule else "Unknown"
        pks_matches = len([n for n in G.nodes if G.nodes[n].get('is_pks_match', False)])
        ax.set_title(
            f"DORAnet MCTS Search Tree\nTarget: {target_smiles}\n"
            f"Total Nodes: {len(G.nodes)}, PKS Matches: {pks_matches}",
            fontsize=12, fontweight='bold'
        )

    # Add stats box
    stats_text = (
        f"MCTS Statistics:\n"
        f"  Iterations: {agent.total_iterations}\n"
        f"  Max Depth: {agent.max_depth}\n"
        f"  Total Nodes: {len(agent.nodes)}\n"
        f"  PKS Library Size: {len(agent.pks_library)}\n"
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
            is_pks_match = agent.calculate_reward(node) > 0
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

    for n in node_ids:
        data = G.nodes[n]
        is_pks = data.get('is_pks_match', False)
        prov = data.get('provenance', 'target')

        if is_pks:
            colors.append('#2ecc71')
        elif prov == 'enzymatic':
            colors.append('#3498db')
        elif prov == 'synthetic':
            colors.append('#9b59b6')
        else:
            colors.append('#f39c12')

        visits = data.get('visits', 1)
        sizes.append(10 + visits * 3)

        smiles_list.append(data.get('smiles', 'N/A'))
        visits_list.append(data.get('visits', 0))
        values_list.append(f"{data.get('avg_value', 0):.3f}")
        provenance_list.append(prov)
        pks_match_list.append('Yes ✓' if is_pks else 'No')

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
    """
    try:
        from bokeh.plotting import figure, save, output_file
        from bokeh.models import HoverTool, ColumnDataSource, LabelSet
        from bokeh.layouts import column
        from bokeh.models.annotations import Title
    except ImportError:
        print("[Visualization] Bokeh not installed. Install with: pip install bokeh")
        return

    from rdkit import Chem

    print("[Visualization] Generating enhanced interactive visualization...")
    print("[Visualization] Creating molecule structure images...")

    # Create graph
    G = create_tree_graph(agent)
    if len(G.nodes) == 0:
        print("[Visualization] No nodes to visualize.")
        return

    pos = get_hierarchical_pos(G, root=0)

    # Prepare node data with molecule images
    node_ids = []
    x_coords = []
    y_coords = []
    colors = []
    sizes = []
    smiles_list = []
    smiles_short_list = []
    visits_list = []
    values_list = []
    avg_values_list = []
    provenance_list = []
    pks_match_list = []
    depth_list = []
    mol_images = []

    for n in G.nodes:
        data = G.nodes[n]
        is_pks = data.get('is_pks_match', False)
        prov = data.get('provenance', 'target')
        smiles = data.get('smiles', '')
        smiles_short = data.get('smiles_short', str(n))

        # Color based on type (same as static visualization)
        if is_pks:
            color = '#2ecc71'  # Green for PKS match
        elif prov == 'enzymatic':
            color = '#3498db'  # Blue for enzymatic
        elif prov == 'synthetic':
            color = '#9b59b6'  # Purple for synthetic
        else:
            color = '#f39c12'  # Orange for target

        visits = data.get('visits', 0)
        value = data.get('value', 0.0)
        avg_value = data.get('avg_value', 0.0)
        depth = data.get('depth', 0)

        # Generate molecule image
        mol_img = _generate_molecule_image_base64(smiles, size=molecule_img_size)

        node_ids.append(n)
        x_coords.append(pos[n][0])
        y_coords.append(pos[n][1])
        colors.append(color)
        sizes.append(min(15 + visits * 3, 50))
        smiles_list.append(smiles if smiles else 'N/A')
        smiles_short_list.append(smiles_short)
        visits_list.append(visits)
        values_list.append(f"{value:.2f}")
        avg_values_list.append(f"{avg_value:.3f}")
        provenance_list.append(prov.capitalize())
        pks_match_list.append('✓ Yes' if is_pks else '✗ No')
        depth_list.append(depth)
        mol_images.append(mol_img if mol_img else "")

    # Create node data source
    node_source = ColumnDataSource(data=dict(
        x=x_coords,
        y=y_coords,
        node_id=node_ids,
        color=colors,
        size=sizes,
        smiles=smiles_list,
        smiles_short=smiles_short_list,
        visits=visits_list,
        value=values_list,
        avg_value=avg_values_list,
        provenance=provenance_list,
        pks_match=pks_match_list,
        depth=depth_list,
        mol_img=mol_images,
    ))

    # Prepare edge data with reaction information
    print("[Visualization] Extracting reaction information for edges...")
    edge_x0 = []
    edge_y0 = []
    edge_x1 = []
    edge_y1 = []
    edge_reactions = []
    edge_colors = []

    for parent_id, child_id in agent.edges:
        if parent_id in pos and child_id in pos:
            # Get child node for reaction info
            child_node = next((n for n in agent.nodes if n.node_id == child_id), None)

            # Extract reaction information
            if child_node:
                rxn_label = child_node.reaction_name or "Unknown reaction"

                # Truncate for tooltip
                rxn_label_short = rxn_label[:100] + "..." if len(rxn_label) > 100 else rxn_label

                reaction_info = rxn_label_short

                # Edge color based on provenance
                if child_node.provenance == 'enzymatic':
                    edge_color = '#3498db'
                elif child_node.provenance == 'synthetic':
                    edge_color = '#9b59b6'
                else:
                    edge_color = '#95a5a6'
            else:
                reaction_info = "No reaction information"
                edge_color = '#95a5a6'

            edge_x0.append(pos[parent_id][0])
            edge_y0.append(pos[parent_id][1])
            edge_x1.append(pos[child_id][0])
            edge_y1.append(pos[child_id][1])
            edge_reactions.append(reaction_info)
            edge_colors.append(edge_color)

    # Create edge data source
    edge_source = ColumnDataSource(data=dict(
        x0=edge_x0,
        y0=edge_y0,
        x1=edge_x1,
        y1=edge_y1,
        reaction=edge_reactions,
        color=edge_colors,
    ))

    # Create Bokeh figure
    target_smiles = Chem.MolToSmiles(agent.target_molecule) if agent.target_molecule else "Unknown"
    pks_matches = len([n for n in G.nodes if G.nodes[n].get('is_pks_match', False)])

    p = figure(
        title=f"DORAnet MCTS Interactive Search Tree",
        width=1400,
        height=900,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom",
    )

    # Add subtitle with target info
    p.add_layout(Title(
        text=f"Target: {target_smiles[:100]} | Total Nodes: {len(G.nodes)} | PKS Matches: {pks_matches}",
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

    # Edge hover tool
    edge_hover = HoverTool(
        renderers=[edge_lines],
        tooltips=[
            ("Reaction", "@reaction"),
        ],
        point_policy="follow_mouse"
    )
    p.add_tools(edge_hover)

    # Draw nodes
    node_circles = p.scatter(
        'x', 'y',
        source=node_source,
        size='size',
        color='color',
        line_color='#2c3e50',
        line_width=2.5,
        alpha=0.9,
        marker='circle'
    )

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
            <b style="color: #2c3e50;">Visits:</b> @visits<br>
            <b style="color: #2c3e50;">Avg Value:</b> @avg_value<br>
            <b style="color: #2c3e50;">SMILES:</b><br>
            <span style="font-size: 10px; word-break: break-all;">@smiles</span>
        </div>
    </div>
    """

    node_hover = HoverTool(
        renderers=[node_circles],
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
        # Note: text_font_weight not supported in LabelSet
    )
    p.add_layout(labels)

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
