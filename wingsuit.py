from typing import TypedDict
import numpy as np
from scipy.integrate import solve_ivp

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

def get_coef_matrix(p: WingsuitProperties, U: float, log = False):
    theta_ddot_coef = p['I'] - p['m'] * p['l_t'] * p['l_t']

    theta_ddot_rhs = np.array([
        p['C_t'] * U * U * (p['l_a'] + p['l_t']) - p['k_t'], # theta coef
        -p['l_t'] * p['k'], # y coef
        -p['d_t'], # theta_dot coef
        p['C_y'] * U * (p['l_a'] + p['l_t']) - p['l_t'] * p['d'] # y_dot coef
    ]) / theta_ddot_coef

    y_ddot_coef = p['m'] - p['m'] * p['m'] * p['l_t'] * p['l_t'] / p['I']

    y_ddot_rhs = np.array([
        p['C_t'] * U * U + p['m'] * p['l_t'] * (p['l_a'] * p['C_t'] * U * U - p['k_t']) / p['I'], # theta coef
        -p['k'], # y coef
        -p['m'] * p['l_t'] * p['d_t'] / p['I'], # theta_dot coef
        p['C_y'] * U * (1 + p['m'] * p['l_t'] * p['l_a'] / p['I']) - p['d'], # y_dot coef
    ]) / y_ddot_coef

    # State structure is:
    # _____________
    # | theta     |
    # | y         |
    # | theta_dot |
    # | y_dot     |
    # _____________

    A = np.row_stack([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        theta_ddot_rhs,
        y_ddot_rhs
    ])

    if(log):
        print(A)
        print(theta_ddot_coef)
        print(y_ddot_coef)

    return A

def get_wingsuit_eigs(p: WingsuitProperties, airspeed_values: np.ndarray[float]):
    all_eigs = [np.linalg.eig(get_coef_matrix(p, U))[0] for U in airspeed_values]
    
    return np.transpose(all_eigs)

def simulate(properties: WingsuitProperties, initial_y_velocity: float, initial_theta_velocity: float, airspeed: float):
    y0 = [0, 0, initial_theta_velocity, initial_y_velocity]
    
    t_span = (0, 1)
    t_eval = np.linspace(*t_span, 100)

    A = get_coef_matrix(properties, airspeed, True)

    return solve_ivp(lambda t, Y: A @ Y, t_span, y0)

