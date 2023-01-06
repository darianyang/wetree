"""
This code takes in a Wepy hdf5 file and generates a gexf file for
visualization of the simulations resampling as a tree.
"""

import numpy as np
import networkx as nx
import sys

from wepy.hdf5 import WepyHDF5
from wepy.analysis import parents
from restree.propagation import generate_node_positions
from restree.parent_tree import ParentForest
from wepy.analysis.parents import resampling_panel, parent_panel
from wepy.resampling.decisions.clone_merge import MultiCloneMergeDecision

# args - hdf5 name and output gexf filename
input_file = sys.argv[1]
outfile_name = sys.argv[2] 

# Load wepy hdf5 file into python script
wepy_h5 = WepyHDF5(input_file, mode = 'r')
wepy_h5.open()

run_idx = 0 
n_walkers = wepy_h5.num_init_walkers(run_idx)
n_cycles = wepy_h5.num_run_cycles(run_idx)

# Make Parent Table
records = wepy_h5.resampling_records([0])
resampling_panel = resampling_panel(records) 
parent_panel = parents.parent_panel(MultiCloneMergeDecision, resampling_panel)
parent_matrix = parents.net_parent_table(parent_panel)
parent_forest = ParentForest(np.array(parent_matrix))

# a matrix of the weights for all the walkers in the run
walker_weights = []
for traj_fields in wepy_h5.iter_trajs_fields(['weights']):
    for field, data in traj_fields.items():
        walker_weights.append(np.array(data))

weight_matrix = np.array(walker_weights).squeeze().T
weight_matrix = np.array(weight_matrix)
weight_list = weight_matrix.tolist() + [weight_matrix[-1].tolist()]
parent_forest.set_step_attrs('weight', weight_list)

# Energy Constants:
R_0 =  1000
B = 0.01
NODE_RADIUS = 10
C = 5
G = 2.5
FAN_SIZE = 1.5
NODE_SIZE = 3

node_positions = generate_node_positions(weight_matrix, parent_matrix,
                                         B, C, G, R_0, NODE_RADIUS, FAN_SIZE, NODE_SIZE)

# rescale them on the y-axis
node_positions[:,:,1] *= NODE_RADIUS * 2
parent_forest.set_step_positions(node_positions)


print('Tree made')

# Tree is made. Most of the rest of the code is for coloring and 
# visualization in gephi. Last line saves the tree.

# Get the data of interest for coloring the graph
# Here it is an observable named 'com_to_com'

walker_com = []
for traj_fields in wepy_h5.iter_trajs_fields(['observables/com_to_com']): 
    for field, data in traj_fields.items(): 
        walker_com.append(data)
com_matrix = np.array(walker_com).squeeze().T 

# Add the COM data to the graph
for cycle in range(n_cycles): 
    for walker in range(n_walkers): 
        parent_forest.nodes[(cycle,walker)]['COM'] = com_matrix[cycle, walker]


# Removes extra line of nodes 
for walker in range(n_walkers):
     parent_forest.remove_node((n_cycles, walker))


nx.write_gexf(parent_forest.to_undirected(), outfile_name)
