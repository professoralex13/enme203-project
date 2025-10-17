from typing import TypedDict
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import linear_sum_assignment

class WingsuitProperties(TypedDict):
    k: float
    k_t: float

    d: float
    d_t: float

    l_t: float
    l_a: float

    m: float
    I: float
    
    C_t: float
    C_y: float

def get_coef_matrix(p: WingsuitProperties, U: float):
    """
        Gets the coefficient matrix for a given wingsuit and airspeed
        which can be used for numerical ODE solving or modal analysis
    """
    M = np.array([
        [p['I'], -p['m'] * p['l_t']],
        [-p['m'] * p['l_t'], p['m']]
    ])

    D = np.array([
        [p['d_t'], -p['l_a'] * p['C_y'] * U],
        [0, p['d'] - p['C_y'] * U]
    ])

    K = np.array([
        [p['k_t'] - p['l_a'] * p['C_t'] * U * U, 0],
        [-p['C_t'] * U * U, p['k']]
    ])

    
    A = np.linalg.inv(np.block([
        [np.identity(2), np.zeros((2, 2))],
        [np.zeros((2, 2)), M]
    ])) @ np.block([
        [np.zeros((2, 2)), np.identity(2)],
        [-K, -D]
    ])

    return A

# Function written by Claude Sonnet 4.5 for the purposes 
# of improving the readability of eigenvalue graphs
#
# The behavior in this function is outside the scope of ENME203
# and does not have an impact on the findings of the project
def sort_eigenvalues_continuous(eigenvalue_sets):
    """
    Sort eigenvalues to maintain continuity 
    across sets and prevent mode jumping.
    
    Parameters:
    -----------
    eigenvalue_sets : array-like, shape (n_sets, 4)
        Array where each row contains 4 complex eigenvalues
    
    Returns:
    --------
    sorted_modes : ndarray, shape (4, n_sets)
        Array where each row contains eigenvalues 
        of the same mode across all sets
    """
    eigenvalue_sets = np.array(eigenvalue_sets)
    n_sets, n_modes = eigenvalue_sets.shape
    
    if n_modes != 4:
        raise ValueError("Each set must contain exactly 4 eigenvalues")
    
    # Initialize output array
    sorted_modes = np.zeros((n_modes, n_sets), dtype=complex)
    
    # First set: sort by some consistent rule 
    # (e.g., by real part, then imaginary)
    first_set = eigenvalue_sets[0]
    idx = np.lexsort((first_set.imag, first_set.real))
    sorted_modes[:, 0] = first_set[idx]
    
    # For each subsequent set, match to previous set
    for i in range(1, n_sets):
        current_set = eigenvalue_sets[i]
        previous_modes = sorted_modes[:, i-1]
        
        # Compute distance matrix between current 
        # eigenvalues and previous modes
        # Using absolute difference in complex plane
        cost_matrix = np.zeros((n_modes, n_modes))
        for j in range(n_modes):
            for k in range(n_modes):
                cost_matrix[j, k] = np.abs(previous_modes[j] - current_set[k])
        
        # Use Hungarian algorithm to find optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Assign eigenvalues to modes based on optimal matching
        sorted_modes[:, i] = current_set[col_ind]
    
    return sorted_modes

def get_wingsuit_eigs(
    p: WingsuitProperties,
    airspeed_values: np.ndarray[float]
):
    """
        Gets sets of the four wingsuit-system 
        eigenvalues for a given set of airspeed values
    """
    all_eigs = np.array([
        np.linalg.eig(get_coef_matrix(p, U))[0] for U in airspeed_values
    ])
    
    return sort_eigenvalues_continuous(all_eigs)

def simulate(
    properties: WingsuitProperties,
    initial_y_velocity: float,
    initial_theta_velocity: float,
    airspeed: float
):
    """
        Models the translational and rotational behavior of a wingsuit 
        over a 1 second period given initial conditions and an airspeed
    """
    y0 = [0, 0, initial_theta_velocity, initial_y_velocity]
    
    t_span = (0, 1)
    t_eval = np.linspace(*t_span, 100)

    A = get_coef_matrix(properties, airspeed)

    return solve_ivp(lambda t, Y: A @ Y, t_span, y0, t_eval=t_eval,)
