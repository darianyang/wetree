
def make_graph_rescale(distance_matrix, weight_matrix, parent_matrix, rmsd_data, positional_data_storage, number_of_timesteps, number_of_walkers):
    FAN_SIZE = 0
    NODE_SIZE = 0
    positional_data = np.linspace(-number_of_walkers+1, number_of_walkers-1, number_of_walkers)
    H = nx.Graph()
    iteration = 0
    ultimate_parent_matrix = make_ultimate_parent_matrix(parent_matrix, number_of_timesteps, number_of_walkers)

    #sum_of_distance_matrix = sum_of_distances_matrix_generator(distance_matrix, number_of_timesteps, number_of_walkers)

    for time_step in range(number_of_timesteps):

        # Writes dictionary to add positional data
        for node_in_step in range(number_of_walkers):
            viz = {"position": {'x': float(positional_data_storage[time_step, node_in_step]),
                            'y': float(time_step),
                            'z': 0}}

            # Adds weighted node to the graph
            H.add_node(iteration, viz=viz,
                       weight= float(weight_matrix[time_step, node_in_step]),
                       ultimate_parent = float(ultimate_parent_matrix[time_step,
                                                                  node_in_step]),
                       #sum_of_walker_distances = float(sum_of_distance_matrix[time_step, node_in_step]),
                       rmsd = float(rmsd_data[time_step, node_in_step]))

            iteration += 1

        if time_step > 0:
            for beads_1 in range(number_of_walkers):
                for beads_2 in range(number_of_walkers):
                    if positional_data[beads_1] == positional_data_previous[int(parent_matrix[time_step,beads_2])]:
                        H.add_edge((time_step - 1) * len(positional_data_storage[0]) +
                                   parent_matrix[time_step, beads_1],
                                   ((time_step) * len(positional_data_storage[0]) + beads_1))

        if time_step < len(positional_data_storage) - 1:
            positional_data_previous = positional_data
            positional_data = update_x_for_next_timestep(positional_data, parent_matrix[time_step +1], FAN_SIZE, NODE_SIZE)


    return H

def make_graph(node_positions, parent_matrix, number_of_walkers, rmsd_matrix):
    FAN_SIZE = 0
    NODE_SIZE = 0
    positional_data = np.linspace(-number_of_walkers+1, number_of_walkers-1, number_of_walkers)

    H = nx.Graph()
    iteration = 0

    #ultimate_parent_matrix = make_ultimate_parent_matrix(parent_matrix, number_of_timesteps, number_of_walkers)

    #sum_of_distance_matrix = sum_of_distances_matrix_generator(distance_matrix, number_of_timesteps, number_of_walkers)

    for time_step in range(number_of_timesteps):

        # Writes dictionary to add positional data
        for node_in_step in range(number_of_walkers):
            viz = {"position": {'x': float(node_positions[time_step, node_in_step]),
                            'y': float(time_step),
                            'z': 0}}

            # Adds weighted node to the graph
            H.add_node(iteration, viz=viz,
                       weight= float(weight_matrix[time_step, node_in_step]),
                       #ultimate_parent = float(ultimate_parent_matrix[time_step,
                       #                                           node_in_step]),
                       #sum_of_walker_distances = float(sum_of_distance_matrix[time_step, node_in_step]),
                       rmsd = float(rmsd_matrix[time_step, node_in_step]))

            iteration += 1


        #if time_step < len(positional_data_storage) - 1:
        #    positional_data_previous = positional_data
        #    positional_data = update_x_for_next_timestep(positional_data, parent_matrix[time_step +1], FAN_SIZE, NODE_SIZE)


    return H
