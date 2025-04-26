from mpi4py import MPI
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np

# setup MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# parameters