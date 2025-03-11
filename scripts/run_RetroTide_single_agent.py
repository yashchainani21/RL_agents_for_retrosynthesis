from bokeh.plotting import figure, show
from bokeh.models import GraphRenderer, StaticLayoutProvider, Circle, MultiLine, HoverTool, ColorBar, LinearColorMapper
from bokeh.transform import linear_cmap
from bokeh.palettes import Viridis256
import networkx as nx
from bokeh.io import output_notebook
output_notebook()

from rdkit import Chem
from RetroTide_agent.node import Node
from RetroTide_agent.mcts import MCTS

root = Node(PKS_product = None,
            PKS_design = None,
            parent = None,
            depth = 0)

mcts = MCTS(root = root,
            target_molecule = Chem.MolFromSmiles("CCCCCC(=O)O"), # OC(CC(O)CC(O)=O)/C=C/C1=CC=CC=C1 # CCCCCC(=O)O # O=C1C=CCC(CO)O1
            max_depth = 3,
            total_iterations = 15000,
            maxPKSDesignsRetroTide = 3000,
            selection_policy = "UCB1")

mcts.run()


