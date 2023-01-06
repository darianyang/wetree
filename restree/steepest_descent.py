import numpy as np
import numpy.linalg as la
import scipy.optimize as sopt

def gradient_energy_function(x, x_previous,
                             distance_matrix, weight_vector,
                             B, C, G, R_0, NODE_RADIUS):

    """This function calculates the energy of the tree at one time step,
    given a move in the system occurs.

    Inputs:

    x(1D numpy array): Indicates positions of the beads at the current
    time step.

    x_previous (1D numpy array): Indicates positions of the beads at
    the previous time step.

    distance matrix (2D numpy array): Matrix that describes the
    distance of the walkers at one time step. Calculated by WExplore.

    weight_vector (1D numpy array) Vector of walker weights. Describes
    the probability of a trajectory being at that node. Calculated
    from WExplore.

    B, C, G (Float) Constants used to scale
    the terms in the energy equation. G is used in the
    Determine Eoij function excuslively.

    Outputs

    energy_new_vector (1D numpy array) An array that contains the
    energy of each bead.

    """

    gradient_vector = np.zeros(len(x))

    for i in range(len(x)):
        partial_derivative_of_energy = 0
        partial_derivative_of_energy += 2 * B * (x[i] - x_previous[i])
        partial_derivative_of_energy += 2 * C * weight_vector[i] * x[i]

        for j in range (len(x)):
            if i == j:
                continue

            e_ij =  determine_eoij(distance_matrix, G, i, j)
            partial_derivative_of_energy += ((4 * e_ij * (x[i] - x[j]) *
                                              np.exp(-(x[i] - x[j]) ** 2 / R_0)) )

            r_ij = x[i] - x[j]

            if r_ij < 2 * NODE_RADIUS:

                partial_derivative_of_energy += NODE_RADIUS / (r_ij ** 13)

        gradient_vector[i] = partial_derivative_of_energy

    return gradient_vector


def energy_function(x, x_previous, distance_matrix, weight_vector, B, C, G, R_0, NODE_RADIUS):

    """This function calculates the energy of the tree at one time step,
    given a move in the system occurs.

    x(1D numpy array): Indicates positions of the beads at the current time step.

    x_previous (1D numpy array): Indicates positions of the beads at
    the previous time step.

    distance matrix (2D numpy array): Matrix that describes the
    distance of the walkers at one time step. Calculated by WExplore.

    Weight_Vector (1D numpy array) Vector of walker weights. Describes
    the probability of a trajectory being at that node. Calculated
    from WExplore.

    B, C, G (Float) Constants used to scale the terms in the energy
    equation. G is used in the Determine Eoij function excuslively.

    Return Energy (Float) The new energy of the system.

    """
    NODE_RADIUS_12 = NODE_RADIUS ** 12

    energy_new = 0
    for i in range (len(x)):
        energy_new += B*(x[i]-x_previous[i])**2
        energy_new += C * weight_vector[i] * x[i] **2
        for j in range(len(x)):
            eoij = determine_eoij(distance_matrix, G, i, j)

            energy_new = energy_new + eoij * np.exp(-(x[i] - x[j]) ** 2 / R_0)

            if i != j:

                r_ij = (x[i] - x[j])

                if r_ij < 2 * NODE_RADIUS:

                    energy_new += NODE_RADIUS_12 / (r_ij ** 12)
    return energy_new


def determine_eoij(distance_matrix, constant, index_1, index_2):
    """Calculates the constant needed to determine the repusion segment
    of the equation.

    Inputs:

    distance matrix (2D numpy array): Matrix that describes the
    distance of the walkers at one time step. Calculated by WExplore.

    constant (Float): Constant used to scale the term of the energy
    equation.

    index_1, index_2 (Float): Constants used to determine which
    distance measurement is required.

    Outputs:

    e_oij (Float): Calculates an energy constant needed to solve the
    energy equation.

    """

    guess = distance_matrix[index_1,index_2] - 2
    if guess > 0:
        e_oij = constant * guess
    else:
        e_oij = 0
    return e_oij


def energy_delta(alpha,
                 x, x_prev,
                 distance_matrix, weight_vector,
                 energy_gradient,
                 B, C, G, R_0, NODE_RADIUS):

    """
    This function calculates the energy from the resultant
    """

    delta = x - alpha * energy_gradient

    return energy_function(delta, x_prev,
                           distance_matrix, weight_vector,
                           B, C, G, R_0, NODE_RADIUS)



def steepest_descent(node_positions, node_positions_previous, distance_matrix,
                    weight_vector, B, C, G, R_0, NODE_RADIUS):

    """This function finds the minimum energy the tree can possess using
    the steepest decent algorithm.

    Inputs:

    node_positions (1D numpy array): Initial positional data for the
    walkers before energy minimization.

    node_positions_previous (1D numpy array): The final positions for
    the walkers after energy minimization.

    distance_matrix (2D numpy array):

    """


    # print('The graph energy is',
    #       energy_function(node_positions_previous_sd, node_positions_previous,
    #                       distance_matrix,
    #                       weight_vector,
    #                       B, C, G, R_0, NODE_RADIUS), '\n')

    # An arbitrary error to ensure the loop is initiaited.
    error = 1
    while error > 0.001:

        energy_gradient = gradient_energy_function(node_positions,
                                                   node_positions_previous,
                                                   distance_matrix,
                                                   weight_vector,
                                                   B, C, G, R_0, NODE_RADIUS)

        # Determines the optimal alpha value to optimize the steepest
        # descent alogrighm.
        alpha_opt = sopt.golden(energy_delta,
                                args = (node_positions, node_positions_previous,
                                        distance_matrix, weight_vector,
                                        energy_gradient,
                                        B, C, G, R_0, NODE_RADIUS))


        # Determines the new walker positions
        node_positions_previous = node_positions
        node_positions =  node_positions_previous - alpha_opt * energy_gradient

        # Calculates the error.
        error_vector = node_positions - node_positions_previous
        error = la.norm(error_vector)

        # print('The graph energy is',
        #       energy_function(node_positions_previous_sd, node_positions_previous,
        #                       distance_matrix,
        #                       weight_vector,
        #                       B, C, G, R_0, NODE_RADIUS), '\n')


    return node_positions
