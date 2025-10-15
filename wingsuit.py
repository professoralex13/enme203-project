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

def get_coef_matrix(p: WingsuitProperties, U: float):
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

    
    A = np.block([
        [np.zeros((2, 2)), np.identity(2)],
        [-K, -D]
    ]) @ np.linalg.inv(np.block([
        [np.identity(2), np.zeros((2, 2))],
        [np.zeros((2, 2)), M]
    ]))

    return A

def get_wingsuit_eigs(p: WingsuitProperties, airspeed_values: np.ndarray[float]):
    all_eigs = [np.linalg.eig(get_coef_matrix(p, U))[0] for U in airspeed_values]
    
    return np.transpose(all_eigs)

def simulate(properties: WingsuitProperties, initial_y_velocity: float, initial_theta_velocity: float, airspeed: float, log = False):
    y0 = [0, 0, initial_theta_velocity, initial_y_velocity]
    
    t_span = (0, 1)
    t_eval = np.linspace(*t_span, 100)


    A = get_coef_matrix(properties, airspeed)

    return solve_ivp(lambda t, Y: A @ Y, t_span, y0, method='BDF',
                t_eval=t_eval,  # This forces output at these times
                rtol=1e-6, 
                atol=1e-9,
                max_step=0.001)
