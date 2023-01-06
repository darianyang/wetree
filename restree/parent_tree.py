import numpy as np
import networkx as nx

class ParentForest(nx.Graph):
    def __init__(self, parent_matrix):
        super().__init__()

        self._n_steps = parent_matrix.shape[0]
        self._n_walkers = parent_matrix.shape[1]

        # make the roots of each tree in the parent graph, the step is
        # 0
        self._roots = [(0, i) for i in range(self.n_walkers)]



        # set these as nodes
        self.add_nodes_from(self.roots)

        # go through the parent matrix and make edges from the parents
        # to children nodes
        for step_idx, parent_idxs in enumerate(parent_matrix):
            step_idx += 1

            # make edge between each walker of this step to the previous step
            for curr_walker_idx in range(self.n_walkers):

                # get the parent index
                parent_idx = parent_idxs[curr_walker_idx]

                # if it is a -1 indicating a discontinuity we add an
                # attribute indicating this is a discontinuity
                discontinuity = False
                if parent_idx == -1:
                    discontinuity = True

                parent_node = (step_idx - 1, parent_idx)
                child_node = (step_idx, curr_walker_idx)

                # make an edge between the parent of this walker and this walker
                edge = (parent_node, child_node)


                self.add_edge(*edge, discontinuous=discontinuity)

    @property
    def roots(self):
        return self._roots

    @property
    def trees(self):
        trees_by_size = [self.subgraph(c) for c in nx.weakly_connected_components(self)]
        trees = []
        for root in self.roots:
            root_tree = [tree for tree in trees_by_size if root in tree][0]
            trees.append(root_tree)
        return trees

    @property
    def n_steps(self):
        return self._n_steps

    @property
    def n_walkers(self):
        return self._n_walkers

    def step(self, step_idx):
        """ Get the nodes at the step (level of the tree)."""

        step_nodes = []
        for node in self.nodes:
            if node[0] == step_idx:
                step_nodes.append(node)

        step_nodes.sort()
        return step_nodes

    def steps(self):
        node_steps = []
        for step_idx in range(self.n_steps + 1):
            node_steps.append(self.step(step_idx))

        return node_steps

    def walker(self, walker_idx):
        """ Get the nodes at the step (level of the tree)."""

        walker_nodes = []
        for node in self.nodes:
            if node[1] == walker_idx:
                walker_nodes.append(node)

        walker_nodes.sort()
        return walker_nodes

    def walkers(self):
        node_walkers = []
        for walker_idx in range(self.n_walkers):
            node_walkers.append(self.walker(walker_idx))

        return node_walkers

    def set_step_attrs(self, key, values):
        """Set attributes on a stepwise basis, i.e. expects a array/list that
        is n_steps long and has the appropriate number of values for
        the number of walkers at each step

        """
        for step in self.steps():
            for node in step:
                self.nodes[node][key] = values[node[0]][node[1]]

    def set_step_positions(self, positions):
        for step in self.steps():
            for node in step:
                coord = positions[node[0]][node[1]]
                viz_dict = {'position' : {'x' : float(coord[0]),
                                           'y' : float(coord[1]),
                                           'z' : float(coord[2])}
                           }

                self.nodes[node]['viz'] = viz_dict


# def make_parent_matrix(resampling_df):
#     ### CAUTION Does not work with all possible resampling records
#     ### supported by WepyHDF5. Conformant parent matrix
#     ### implementation in the wepy module.

#     # Sorts dataframe by cycle index and by step index
#     resampling_df.sort_values(by=['cycle_idx','step_idx'], inplace=True)

#     # Gets the number of walkers and time steps
#     n_cycles = max(resampling_df['cycle_idx']) + 1
#     n_walkers = max(resampling_df['walker_idx']) + 1

#     # Makes a 2D numpy array that will turn into the parent matrix
#     parent_matrix = np.zeros([n_cycles, n_walkers])

#     # Sets up the first row of the parent matrix
#     for walker in range(n_walkers):
#         parent_matrix[0, walker] = walker

#     next_step_walker_indicies = np.copy(parent_matrix[0])

#     time_step = 0
#     for index, row in resampling_df.iterrows():
#         if time_step != ro

#         w['cycle_idx']:
#             parent_matrix[time_step + 1] = np.copy(next_step_walker_indicies)
#             next_step_walker_indicies = np.copy(parent_matrix[0])

#         time_step = row['cycle_idx']
#         walker = row['walker_idx']

#         if row['decision_id'] == 2:
#             for new_index in range(len(row['instruction_record'])):
#                 next_step_walker_indicies[row['instruction_record'][new_index]] = walker

#     return parent_matrix
