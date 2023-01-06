"""
This code takes in a Wepy hdf5 file and generates a gexf file for
visualization of the simulations resampling as a tree.
"""

import numpy as np
import networkx as nx
import h5py
import sys

import wedap

from restree.propagation import generate_node_positions
from restree.parent_tree import ParentForest

# args - hdf5 name and output gexf filename
input_file = sys.argv[1]
outfile_name = sys.argv[2] 

# Load wepy hdf5 file into python script
h5 = h5py.File(input_file, mode="r")

run_idx = 0 

# TODO: get n_iwalkers and n_iters from westpa.h5
n_particles = h5["summary"]["n_particles"]
# note this is **initial** walkers (I may need n_particles per iter)
#n_walkers = wepy_h5.num_init_walkers(run_idx)

# n_iterations
# default to last
last_iter = None # TODO: make arg?
if last_iter is not None:
    n_iters = last_iter
elif last_iter is None:
    n_iters = h5.attrs["west_current_iteration"] - 1
#n_cycles = wepy_h5.num_run_cycles(run_idx)

print(n_iters, n_particles)

# Make Parent Table
# TODO: get parent matrix equivalent from westpa.h5
parent_matrix = []
#for iter in range(self.first_iter, self.last_iter + 1):
# have to make array start from iteration 1 to index well during weighting
# but only for using skipping basis
for iter in range(1, n_iters + 1):
    parent_matrix.append(h5[f"iterations/iter_{iter:08d}/seg_index"]["parent_id"])
# 1D array of variably shaped arrays
parent_matrix = np.array(parent_matrix, dtype=object)
print(parent_matrix)
print(parent_matrix.shape)

# initializes an empty graph object to be filled in later
#parent_forest = ParentForest(np.array(parent_matrix))
#print(parent_forest)

# converted to an array, note I will likely need to use np object format array since the
# number of walkers is not constant per iteration like with revo
# get weight array using wedap
pdist = wedap.H5_Pdist(input_file)
weight_matrix = pdist.weights
print(weight_matrix)
print(weight_matrix.shape)
sys.exit(0)

# TODO: why is this step needed? later this extra line of nodes is deleted
# tested it and indeed we need a n_iter+1 size list to go into set_step_attrs
# maybe because the steps are between iterations, so n_iterations+1 steps = n_iterations
weight_list = weight_matrix.tolist() + [weight_matrix[-1].tolist()]
#print(len(weight_list))
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
# com_matrix is n_iter by n_walker array (note this will also be asymmetric with westpa.h5)
#print(com_matrix)
print("COM matrix: ", com_matrix.shape)

# Add the COM data to the graph
for cycle in range(n_cycles): 
    for walker in range(n_walkers): 
        parent_forest.nodes[(cycle,walker)]['COM'] = com_matrix[cycle, walker]

# Removes extra line of nodes 
for walker in range(n_walkers):
     parent_forest.remove_node((n_cycles, walker))

sys.exit(0)
nx.write_gexf(parent_forest.to_undirected(), outfile_name)
