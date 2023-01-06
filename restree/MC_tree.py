"""
This code takes the results from a W-Explore molecular dynamics simulation and
visualizes the results in a tree network. The output can be visualized in Gephi
software. The nodes resemble the walkers at each time step of the simulation,
and the size of the node resembles the probability the walker occurs at
that specific time step. The edges represent where that walker originated
from at the previous time step. When the walker has is connected to several
other nodes at the next time step, that node was cloned in the simulation,
and when a node is not connected to another node in the next time step,
that walker was merged in with another walker.

You need to type the files locations in the order:

distance_matrix_file, wht_matrix_file, parent_matrix_file

into the command line.

You can optionally type the file locations for the
output files in the order:

x_simulation_data_frame csv file, tree plot gexf file

into the command prompt. If you choose not to, the program has default names
for these outputs.

Inputs:

distance_matrix_file (CSV file): A matrix consisiting of the distance at
each timestep. Since there are 48 walkers, the distance matrix is 48x48.
Each set of 48 rows consist of one distance matrix, and there are several
stacked on top of each other.

wht_matrix_file (CSV file): A file containing an array with the weights
of each walker in the simulation. Each row represents a time step, and
the elements in a row represent the weight of that walker.

parent_matrix_file (CSV file): A file containing an array describing how
the nodes are connected to each other. Each row represents a time step and
each element in the row indicates the index the walker originated from
in the previous time step.

rmsd_matrix_file (CSV file): A file containing the rmsd distances that
describe how close the ligand is to the drug. Used to identify walkers
in the GEFX graph.

Outputs:

x_simulation_data_frame (CSV file): A file containing the positions of the
nodes at a given time step. Each row represents a time step, and the element
in a row represents the x coordinate of a walker in the tree graph.

tree_plot (gexf file): A file which encodes the visualization of the tree
network. It contains the positions, and weights of each walker. It also
contains the edges to visualize the node merging and cloning phenomenon.

Script written by: Tom Dixon

Last update: 06/22/2017
"""
import sys
import click

import numpy as np
import pandas as pd
import networkx as nx



# Constants
DELTA_1 = 0.1
DELTA_2 = 0.1
R_0 = 50
B = 5
C = 1
G = 1
BETA = 1
MAX_DISTANCE = 10
FAN_SIZE = 1.5
NODE_SIZE = 3


def pick_a_way_to_move():

    """
    This function determines how the beads could move.
    When movememnt_type = 1, one bead will move a set distance.
    If movement_type = 2, a group of beads will move a set distance.
    If movement_type = 3, two beads will switch in the graph.

    Inputs:

    None

    Outputs:

    movement_type (float): A pesudo random selection of which movement will
    occur at a given time step.
    """

    random_number = np.random.random()
    if random_number < 0.33:
        movement_type = 1
    elif random_number > 0.67:
        movement_type = 3
    else:
        movement_type = 2
    return (movement_type)


def energy_function(x, x_previous, distance_matrix, weight_vector,
                    constant_1, constant_2, constant_3, r_0):

    """
    This function calculates the energy of the tree at one time step, given a
    move in the system occurs.

    x(1D numpy array): Indicates positions of the beads at the current
    time step.

    x_previous (1D numpy array): Indicates positions of the beads at the
    previous time step.

    distance matrix (2D numpy array): Matrix that describes the distance of the
    walkers at one time step. Calculated by WExplore.

    weight_vector (1D numpy array) Vector of walker weights. Describes the
    probability of a trajectory being at that node. Calculated from WExplore.

    Constant_1, Constant_2, Constant_3 (float) Constants used to scale the
    terms in the energy equation. Constant_3 is used in the determine eoij
    function excuslively.

    Return energy (float) The new energy of the system.
    """

    energy_new = 0

    for i in range(len(x)):
        energy_new = energy_new + constant_1 * (x[i] - x_previous[i]) ** 2
        energy_new += constant_2 * weight_vector[i] * x[i] ** 2

        for j in range(len(x)):
            eoij = determine_eoij(distance_matrix, constant_3, i, j)
            energy_new = energy_new + eoij * np.exp(-(x[i] - x[j]) ** 2 / r_0)

    return energy_new


def determine_eoij(distance_matrix, constant, index_1, index_2):

    """
    Calculates the constant needed to determine the repusion segment of the
    equation.

    Inputs:

    distance matrix (2D numpy array): Matrix that describes the distance of the
    walkers at one time step. Calculated by WExplore.

    constant (float): Constant used to scale the term of the energy equation.

    index_1, index_2 (float): Constants used to determine which distance
    measurement is required.

    Outputs:

    e_oij (float): Calculates an energy constant needed to solve the energu
    equation.
    """

    guess = distance_matrix[index_1, index_2] - 2

    if guess > 0:
        e_oij = constant * guess
    else:
        e_oij = 0

    return e_oij


def update_x_for_next_timestep(x, parent_vector, fan_size, node_size):

    """
    This function will update x, the locations of the beads at the current
    time step to the next time step.

    Inputs:

    x(1D numpy array): Indicates positions of the beads at the current
    time step.

    parent_vector (1D numpy array): A vector that shows the relationship
    between the current and next time step.

    Outputs:

    x_updated (1D numpy array): An updated version of x to be used for the next
    time iteration.
   """

    x_updated = np.zeros(len(x))

    x_copy = np.copy(x)

    for i in range(len(x)):
        x_updated[i] = x_copy[parent_vector[i]]

    """
    This part of the code looks for the same numbers in the vector, and changes
    the distance by 0.05 increments, to get the branch effect. First value goes
    to the left of the center, the next value goes to the right of the center.
    """

    for element_1 in range(len(x_updated)):
        counter_1 = 0
        counter_2 = 0

        while has_overlap(x_updated, element_1, node_size):
            counter_2 += 1
            counter_1 += 1
            if counter_1 % 2 == 0:
                x_updated[element_1] = (x_updated[element_1] + counter_2 *
                                        fan_size)
            else:
                x_updated[element_1] = (x_updated[element_1] - counter_2 *
                                        fan_size)

    return x_updated


def has_overlap(x_updated, element_1, node_size):

    """
    This function tests to see if there is any overlap in the beads during
    a move.

    Inputs:

    x_updated (1D numpy array): An updated version of x to be used for the
    next time iteration.

    elemet_1 (float): A number describing the index of the bead that was just
    moved.

    node_size (float): Describes the minimum distance two nodes can be from
    each other.

    Output:

    True or False statement : Returns True if there is overlap between the
    nodes and False if there are no overlaps.
    """

    for element_2 in range(len(x_updated)):
        if element_1 != element_2:
            if np.abs(x_updated[element_1] - x_updated[element_2]) < node_size:
                return True
    return False


def move_single_bead(x, delta_1, node_size):

    """
    This function finds one random point in the x vector, and moves it by a
    set amount, delta_1.

    Inputs:

    x(1D numpy array): Indicates positions of the beads at the current
    time step.

    delta_1 (float): A scalar distance to move a single point x.

    Outputs:

    x(1D numpy array): Indicates positions of the beads at the current
    time step.

    """

    x_single_move = np.copy(x)

    index = np.random.randint(0, len(x))  # Picks a bead
    pick_direction = np.random.random()  # Picks a direction

    too_close = True

    if pick_direction > 0.5:            # Moves bead
        while too_close:
            x_single_move[index] = x_single_move[index] + delta_1
            too_close = has_overlap(x_single_move, index, node_size)

    else:
        while too_close:
            x_single_move[index] = x_single_move[index] - delta_1
            too_close = has_overlap(x_single_move, index, node_size)

    return (x_single_move)


def move_multiple_beads(x, delta_2=DELTA_2):

    """
    This function finds one random point in the x vector, and moves it and all
    points after the bead by a set amount, delta_2.

    Inputs:

    x(1D numpy array): Indicates positions of the beads at the current
    time step.

    delta_2 (float): A scalar distance to move a single point x.

    Outputs:

    x(1D numpy array): Indicates positions of the beads at the current
    time step.
    """

    x_multiple_move = np.copy(x)

    # Finds the max and min distances
    min_x = np.min(x_multiple_move)
    max_x = np.max(x_multiple_move)

    # Ramdomly determines a position to move
    random_position = (max_x - min_x) * np.random.random() + min_x

    # Decides to move the beads to the left or right
    pick_direction = np.random.random()
    for x_index in range(0, len(x)):  # Moves the beads
        if pick_direction > 0.5:
            if x_multiple_move[x_index] > random_position:
                x_multiple_move[x_index] = x_multiple_move[x_index] + delta_2
        else:
            if x_multiple_move[x_index] < random_position:

                x_multiple_move[x_index] = x_multiple_move[x_index] - delta_2

    return (x_multiple_move)


def switch_beads(x, max_distance):

    """
    This function switches two points in the x_vector.

    Inputs:

    x(1D numpy array): Indicates positions of the beads at the current
    time step.

    max_distance (float): An indicator of the maximum distance two beads
    can be in order to switch.

    Outputs:

    x_switched (1D numpy array): Indicates positions of the beads at the
    current time step after switching.
    """
    too_far = True

    while too_far:

        index_1 = np.random.randint(0, len(x))
        direction = np.random.random()

        # An arbitrary large number
        closest_distance = 10000000000000000000000

        if direction > 0.5:
            # Pick a bead right
            for x_index in range(0, len(x)):
                if x[x_index] > x[index_1]:
                    distance = abs(x[x_index] - x[index_1])
                    if distance < closest_distance and (distance <
                                                        max_distance):
                        closest_distance = distance
                        index_2 = x_index

            if closest_distance < max_distance:
                too_far = False

        else:
            # Pick a bead to the left.
            for x_index in range(0, len(x)):
                if x[x_index] < x[index_1]:
                    distance = abs(x[x_index] - x[index_1])
                    if distance < closest_distance and (distance <
                                                        max_distance):

                        closest_distance = distance
                        index_2 = x_index

            if closest_distance < max_distance:
                too_far = False

    # Ensures that two different indicies are selected
    x_switch = np.copy(x)

    # Switches the two beads
    x_switch[index_1] = x[index_2]
    x_switch[index_2] = x[index_1]

    return (x_switch)


def choose_to_accept_new_energy(energy_old, energy_new, beta):

    """
    This function determines if the move in the graph should be accepted or
    rejected.

    Inputs:

    energy_old (float): This is a number that describes the energy of the
    system at the previous time step.

    energy_new (float): This is a number that describes the energy of the
    system at the current time step.

    Outputs:

    energy (float): This is a number that describes the selected energy of the
    system at the current time step.
    """

    # Accepts new energy if the new energy is lower than the old energy
    if energy_new < energy_old:
        energy = energy_new

    else:
        a = np.random.random()

        # Accepts a higher energy system
        if a < np.exp(-beta * (energy_new - energy_old)):
            energy = energy_new
        else:
            # Rejects energy otherwise
            energy = energy_old

    return energy


def single_time_step(x_original, x_previous, distance_matrix, weight_vector,
                     delta_1, delta_2, b, c, g, r_0, beta, max_distance,
                     node_size, eij_old, energy_counter, single_counter,
                     multiple_counter, switch_counter):

    """
    This function moves the bead(s) and determines the new energy of the node
    graph. If the energy is lower, the new map is accepted, if not, then
    there is a small chance that it is accepted.

    Inputs:

    x_original (1D numpy array): The original node map at a given time step
    before the nodes move.

    x_previous (1D numpy array): The final node map at the previous time step.

    distance_matrix (2D numpy array): The distances between each node for a
    given time step. Produced by WExplore simulation.

    weight_vector (1D numpy array): The weights of each node for a given
    time step. Produced by WExplore simulation.

    delta_1 (float): A constant that describes how far the node moves if a
    single is moved node.

    delta_2 (float): A constant that describes how far the nodes moves if
    multiple nodes are moved at once.

    b, c, g, r_0  (float): Constants used to scale the terms in the energy
    equation. g is used in the Determine Eoij function excuslively.

    beta (float): A constant used to determine if the new energy is accepted if
    the new energy is higher than the original energy.

    max_distance (float): An indicator of the maximum distance two beads
    can be in order to switch.

    eij_old (float): The energy of the system before the node locations are
    moved.

    energy_counter (float): The number of times the energy has been switched in
    a given time step.

    Outputs:

    x_new (1D numpy array): The new node map at a given time step after the
    nodes move.

    energy_new (float): This is a number that describes the energy of
    the system after the nodes move.

    energy_counter (float): The number of times the energy has been
    switched in a given time step.
    """
    # Decides which type of move to do.
    bead_move = pick_a_way_to_move()

    # Performs bead movement
    if bead_move == 1:
        x_predicted = move_single_bead(x_original, delta_1, node_size)

    elif bead_move == 2:
        x_predicted = move_multiple_beads(x_original, delta_2)

    else:
        x_predicted = switch_beads(x_original, max_distance)

    # Calculates energy based on new bead location
    eij_new_guess = energy_function(x_predicted, x_previous, distance_matrix,
                                    weight_vector, b, c, g, r_0)

    # Decides if we should accept or reject the New energy
    energy_new = choose_to_accept_new_energy(eij_old, eij_new_guess, beta)

    # Update Gameboard if a different energy was selected.
    if energy_new != eij_old:
        x_new = x_predicted
        energy_counter = energy_counter+1

        if bead_move == 1:
            single_counter += 1

        elif bead_move == 2:
            multiple_counter += 1

        else:
            switch_counter += 1

    else:
        x_new = x_original

    return(x_new, energy_new, energy_counter, single_counter,
           multiple_counter, switch_counter)


def init_walker_positions(n_walkers):
    x = np.linspace(0.01*(-n_walkers+1),
                    0.01*(n_walkers-1), n_walkers)

    return x

def monte_carlo_minimization(parent_matrix, distance_array, weight_matrix, minimization_steps,
                             debug=False,
                             b=B, c=C, g=G, r_0=R_0, delta_1=DELTA_1, delta_2=DELTA_2,
                             beta=BETA, max_distance=MAX_DISTANCE, fan_size=FAN_SIZE,
                             node_size=NODE_SIZE):

    energy_change_list = []
    single_move_list = []
    multiple_move_list = []
    switch_list = []

    n_iterations = distance_array.shape[0]
    n_walkers = distance_array.shape[1]
    # initialize the matrices
    x_simulations = np.zeros([n_iterations, n_walkers])
    x = init_walker_positions(n_walkers)

    n_iterations = x_simulations.shape[0]

    # energy minimization and movements
    for iterations, distance_matrix in enumerate(distance_array):
        energy_counter = 0
        single_counter = 0
        multiple_counter = 0
        switch_counter = 0

        x_copy = np.copy(x)
        x_simulations[iterations] = x_copy

        # if this is the last iteration
        if iterations == len(weight_matrix) - 1:
            # calculate the starting energy of the graph
            eij_old = energy_function(x, x_previous, distance_matrix,
                                      weight_matrix[iterations], b, c, g, r_0)

            # monte carlo steps
            for single_time_step_iteration in range(0, minimization_steps):
                # modify the same values
                (x_copy, eij_old, energy_counter, single_counter, multiple_counter,
                 switch_counter) = single_time_step(x_copy, x_previous,
                                                    distance_matrix,
                                                    weight_matrix[iterations],
                                                    delta_1, delta_2, b, c, g,
                                                    r_0, beta, max_distance,
                                                    node_size, eij_old,
                                                    energy_counter, single_counter,
                                                    multiple_counter,
                                                    switch_counter)
            # save final result
            x_simulations[iterations] = x_copy

        # if it is not the first one
        elif iterations != 0:

            # Determines the initial energy of the time step
            eij_old = energy_function(x, x_previous, distance_matrix,
                                      weight_matrix[iterations], b, c, g, r_0)

            for single_time_step_iteration in range(0, minimization_steps):

                # Performs 1 iteration to minimize the energy for 1 time step.
                (x_copy, eij_old, energy_counter, single_counter, multiple_counter,
                 switch_counter) = single_time_step(x_copy, x_previous,
                                                    distance_matrix,
                                                    weight_matrix[iterations],
                                                    delta_1, delta_2, b, c, g,
                                                    r_0, beta, max_distance,
                                                    node_size, eij_old,
                                                    energy_counter, single_counter,
                                                    multiple_counter,
                                                    switch_counter)

            if debug:
                print(iterations)
                print(eij_old)
                print('The energy monte carlo accepted moves',
                      (energy_counter / minimization_steps) * 100, '% of the time')
                print('Single beads moved', single_counter, 'times ')
                print('Multiple beads moved', multiple_counter, 'times ')
                print('Beads switched', switch_counter, 'times \n')
                energy_change_list.append((energy_counter / minimization_steps) * 100)
                single_move_list.append(single_counter)
                multiple_move_list.append(multiple_counter)
                switch_list.append(switch_counter)

        # update the walkers
        if iterations != len(weight_matrix) - 1:

            # Stores the simulation data
            x_simulations[iterations] = x_copy

            # updates the previous bead data for the next time step.
            x_previous = x_copy

            # updates the bead data for the next time step
            x = update_x_for_next_timestep(x_copy, parent_matrix[iterations+1],
                                           fan_size, node_size)

    if debug:
        print(np.mean(energy_change_list[1:]))
        print(np.std(energy_change_list[1:]))
        print(np.max(energy_change_list[1:]))
        print(np.min(energy_change_list[1:]))

    return x_simulations

def make_graph(parent_matrix, x_positions, **kwargs):

    n_timesteps = parent_matrix.shape[0]
    n_walkers = parent_matrix.shape[1]

    fan_size = 0
    node_size = 0
    x = np.linspace(-n_walkers+1, n_walkers-1, n_walkers)
    H = nx.Graph()
    node_count = 0
    for time_step in range(len(x_positions)):

        # Writes dictionary to add positional data
        for node_in_step in range(n_walkers):

            # in gexf format
            viz = {"position": {'x': float(x_positions[time_step, node_in_step]),
                                'y': 10 * float(time_step)}}

            # for other data arrays in kwargs
            node_attrs = {}
            for key_name, matrix in kwargs.items():
                value = float(matrix[time_step, node_in_step])
                node_attrs[key_name] = value

            # create the node
            H.add_node(node_count,
                       viz=viz,
                       **node_attrs)

            node_count += 1

        # make the edges
        if time_step > 0:
            for beads_1 in range(n_walkers):
                for beads_2 in range(n_walkers):
                    if x[beads_1] == x_previous[parent_matrix[time_step,beads_2]]:
                        H.add_edge((time_step - 1) * len(x_positions[0]) +
                               parent_matrix[time_step, beads_1],
                               ((time_step) * len(x_positions[0]) + beads_1))

        if time_step < len(x_positions) - 1:
            x_previous = x
            x = update_x_for_next_timestep(x, parent_matrix[time_step +1], fan_size,
                                           node_size)

    return H

def fix_distance_matrix(distance_matrix):
    """ For distance matrices from 2D plaintext arrays we want to convert to a 3D array"""

    dist_array = []
    for i in range(n_timesteps):
        dist_matrix = distance_matrix[i*n_walkers : (i+1)*n_walkers]
        dist_array.append(dist_matrix)
    distance_matrix = np.array(dist_array)

    return distance_matrix


if __name__ == "__main__":


    if len(sys.argv) < 5 or len(sys.argv) > 7:
        print("You need to enter the correct number of inputs \n")
        print("The first input is the distance matrix file location. \n")
        print("The second input is the weight matrix file location. \n")
        print("The third input is the parents matrix file location \n")
        print("The fourth input is a file similar to the weight matrix \n")
        print ("that contains a measure such as SASA or to color the \n")
        print(" nodes in the gephi graph. \n")
        print("There are two add optional outputs you can enter too \n")
        print("The first output is the node positional data csv file location")
        print("to make the tree map. \n")
        print("The second output is the file location for the gfex file")
        print("which contains the coding to produce the tree network plot.")
        quit()

    elif len(sys.argv) == 5:
        distance_matrix_file = sys.argv[1]
        weight_matrix_file = sys.argv[2]
        parent_matrix_file = sys.argv[3]
        rmsd_matrix_file = sys.argv[4]
        x_simulation_file = "x_simulation_data.csv"
        tree_plot_file = "Tree_Plot.gexf"

    elif len(sys.argv) == 6:
        distance_matrix_file = sys.argv[1]
        weight_matrix_file = sys.argv[2]
        parent_matrix_file = sys.argv[3]
        rmsd_matrix_file = sys.argv[4]
        x_simulation_file = sys.argv[5]
        tree_plot_file = "Tree_Plot.gexf"

    else:
        distance_matrix_file = sys.argv[1]
        weight_matrix_file = sys.argv[2]
        parent_matrix_file = sys.argv[3]
        rmsd_matrix_file = sys.argv[4]
        x_simulation_file = sys.argv[5]
        tree_plot_file = sys.argv[6]

    # Loads the data from the WExplore simulation
    # A matrix consisiting of the distance at each timestep.
    # fix a 2D array with 3D data into a 3D array
    distance_matrix = fix_distance_matrix(np.loadtxt(distance_matrix_file))

    # The probability of a ligand-protein conformation being formed.
    # Each row represents a timestep.
    weight_matrix = np.loadtxt(weight_matrix_file)

    # A way to measure distance between the protein and the ligand.
    # Done either by RMSD or SASA measurements.
    rmsd_matrix = np.loadtxt(rmsd_matrix_file)

    # A vector to keep track of cloning and merging. Each row represents
    # a time step. The elements in each row indicate the index that walker
    # came from in the previous row.
    parent_matrix = np.loadtxt(parent_matrix_file, dtype=int)


    # initialize the positions
    # Defines the initial values for the starting locations of the walkers.
    # x is the walker positions vector

    n_walkers = len(distance_matrix.shape[1])
    n_timesteps = len(distance_matrix.shape[0])

    x = init_walker_positions(n_walkers, n_timesteps)

    # minimize the energies
    x_simulations = monte_carlo_minimization(parent_matrix, distance_matrix, weight_matrix,
                                             2000,
                                             debug=True)

    # I/O
    # raw walker node positions
    # Writes positional data into a text file.
    x_simulation_data_frame = pd.DataFrame(x_simulations)
    x_simulation_data_frame.to_csv('make_tree_plot.csv', index=False, header=False)

    # make the graph
    # Defines the initial values for the starting locations of the walkers.

    H = make_graph(parent_matrix, x_simulations,
                    weight=weight_matrix,
                    rmsd=rmsd_matrix)

    # Back to I/O
    # Outputs graph to gefx file.
    nx.write_gexf(H, tree_plot_file)
