import numpy as np

from restree.steepest_descent import steepest_descent

def propagate_next_step(x, parent_vector, FAN_SIZE, NODE_SIZE):
    """This function will update x, the locations of the beads at the
    current timestep to the next time step.

    Inputs:

    x(1D numpy array): Indicates positions of the beads at the current
    time step.

    Parent_Vector (1D numpy array): A vector that shows the
    relationship between the current and next time step.

    Outputs:

    x_updated (1D numpy array): An updated version of x to be used for
    the next time iteration.

    """

    x_updated = np.zeros(len(x))

    for i in range (x.shape[0]):
        x_updated[i] = x[parent_vector[i]]

    # This part of the code looks for the same numbers in the vector,
    # and changes the distance by 0.05 increments, to get the branch
    # effect. First value goes to the left of the center, the next
    # value goes to the right of the center.
    for element_1 in range(len(x)):
        counter_1 = 0
        counter_2 = 0

        while has_overlap(x_updated, element_1, NODE_SIZE) == True:
            counter_2 += 1
            counter_1 += 1
            if counter_1 % 2 == 0:
                x_updated[element_1] = x_updated[element_1] + counter_2 * FAN_SIZE
            else:
                x_updated[element_1] = x_updated[element_1] - counter_2 * FAN_SIZE

    return x_updated

def has_overlap (positional_data, element_1, NODE_SIZE):

    """
    This function determines if the walkers are too close together on the graph.

    Inputs:

    positional_data (1D numpy array): An array containing walker positional information
    at the current time step.

    element_1 (float): The specific walker being tested to see if it is too close
    to any other walkers.

    NODE_SIZE (float): The minimum distance two walkers can be to each other.

    Output:

    True/False: If the walker being tested is too close, returns a true statement,
    if not, it returns a false statement.
    """

    for element_2 in range(len(positional_data)):
        if element_1 != element_2:
            if np.abs(positional_data[element_1] - positional_data[element_2]) < NODE_SIZE:
                return True
    return False

def generate_node_positions(weight_matrix, parent_matrix,
                            B=0.01, C=5, G=2.5, R_0=1000,
                            NODE_RADIUS=3, FAN_SIZE=1.5, NODE_SIZE=3):

    n_timesteps = weight_matrix.shape[0]
    n_walkers = weight_matrix.shape[1]

    cycles_node_positions = np.zeros((n_timesteps+1, n_walkers, 3))

    # go through the walker nodes for each cycle and perform
    # minimization

    # initialize the first cycle node positions
    node_positions = np.linspace(0.01 * (-n_walkers + 1), 0.01 * (n_walkers - 1), n_walkers)

    # save them as full coordinates for visualization
    cycles_node_positions[0] = np.array([np.array([x, 0.0, 0.0])
                                         for x in node_positions])

    # propagate nodes for the second step, this will seed the process
    # which can be done iteratively following this
    node_positions_previous = node_positions
    node_positions = propagate_next_step(node_positions, parent_matrix[0],
                                         FAN_SIZE, NODE_SIZE)

    # propagate and minimize the rest of the step nodes
    for step_idx in range(0, n_timesteps):

        weight_vector = weight_matrix[step_idx]

        # add in the other dimensions, y (step_idx) and z (nothing)
        # for the node positions and save them for output
        cycles_node_positions[step_idx+1] = np.array([np.array([x, float(step_idx+1), 0.0])
                                                    for x in node_positions])


        # continue with propagating positions (along x) for the next steps
        node_positions_previous = node_positions
        if step_idx != n_timesteps - 1:
            node_positions = propagate_next_step(node_positions,
                                                      parent_matrix[step_idx + 1],
                                                      FAN_SIZE, NODE_SIZE)

    return cycles_node_positions
