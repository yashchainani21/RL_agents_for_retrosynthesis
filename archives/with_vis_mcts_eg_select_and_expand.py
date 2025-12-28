import networkx as nx
from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxZoomTool, ResetTool
from bokeh.models.graphs import from_networkx
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LabelSet
from rdkit import Chem
from RetroTide_agent.node import Node
from RetroTide_agent.mcts import MCTS

root = Node(PKS_product = None,
            PKS_design = None,
            parent = None,
            depth = 0)

mcts = MCTS(root = root,
            target_molecule = Chem.MolFromSmiles("CCCCC"),
            max_depth = 10,
            maxPKSDesignsRetroTide = 25,
            selection_policy = "UCB1")

selected_node = mcts.select(node = root)
mcts.expand(node = selected_node)
mcts.expand(node = root.children[0])

print(root.children[0])
print('')
print(root.children[1])
print('')
print(root.children[2])

def visualize_mcts_tree(root_node):
    # Create a new directed graph
    G = nx.DiGraph()

    # This recursive function will populate the graph G and node_info dictionary
    def build_graph(node, depth = 0, parent_id=None):

        node_colors = ['red','green','blue','grey']

        # We use id(node) as unique identifier for the node in the graph
        node_id = id(node)
        G.add_node(node_id,
                   depth = depth,
                   visits = node.visits,
                   value = node.value,
                   node_color = node_colors[depth])

        if parent_id is not None:
            G.add_edge(parent_id, node_id)

        for child in node.children:
            build_graph(child, depth + 1, node_id)

    # Populate the graph starting from the root node
    build_graph(root_node)

    # Convert our graph into a Bokeh-compatible data source
    graph_renderer = from_networkx(G, nx.spring_layout, scale=10, center=(0, 0))
    graph_renderer.node_renderer.glyph = Circle(size=15, fill_color = "node_color")

    # Create Bokeh plot
    plot = figure(title="MCTS Tree Visualization", tools="", toolbar_location=None)

    # Define hover tool tips to display information about each node
    hover = HoverTool(tooltips=[("Visits", "@visits"),
                                ("Value", "@value"),
                                ("Depth","@depth")])
    plot.add_tools(hover)

    # Draw the network
    plot.renderers.append(graph_renderer)

    # Display the plot
    show(plot)

visualize_mcts_tree(root)
