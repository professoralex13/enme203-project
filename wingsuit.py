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

    return A

def identify_and_sort_modes(eigenvalues):
    """
    Separate eigenvalues into two physical modes.
    Returns: (mode_A_eigs, mode_B_eigs)
    Each is either a complex number or pair of reals
    """
    # Separate complex and real eigenvalues
    complex_eigs = [e for e in eigenvalues if abs(e.imag) > 1e-10]
    real_eigs = [e for e in eigenvalues if abs(e.imag) <= 1e-10]
    
    modes = []
    
    # Group complex conjugate pairs
    processed = set()
    for i, e in enumerate(complex_eigs):
        if i in processed:
            continue
        # Find conjugate
        for j, e2 in enumerate(complex_eigs):
            if j != i and abs(e - np.conj(e2)) < 1e-10:
                processed.add(i)
                processed.add(j)
                # Take the one with positive imaginary part
                modes.append(e if e.imag > 0 else e2)
                break
    
    # Add real eigenvalues (these are separate modes when uncoupled)
    # For tracking, take the most unstable real eigenvalue for each "mode"
    if len(real_eigs) > 0:
        # If we already have some modes, remaining reals go with closest mode
        # Otherwise, group them
        if len(modes) < 2:
            # Take two most unstable (largest real part)
            real_eigs_sorted = sorted(real_eigs, key=lambda x: x.real, reverse=True)
            for e in real_eigs_sorted[:2-len(modes)]:
                modes.append(e)
    
    return modes

def get_modal_properties(p: WingsuitProperties, airspeed_values: np.ndarray[float]):
    all_eigs = [np.linalg.eig(get_coef_matrix(p, U))[0] for U in airspeed_values]
    
    # Track modes across airspeed
    mode_a = []
    mode_b = []
    
    for i, eigs in enumerate(all_eigs):
        modes = identify_and_sort_modes(eigs)
        
        if i == 0:
            # Initialize based on frequency
            modes_sorted = sorted(modes, key=lambda x: abs(x.imag), reverse=True)
            mode_a.append(modes_sorted[0] if len(modes_sorted) > 0 else 0)
            mode_b.append(modes_sorted[1] if len(modes_sorted) > 1 else 0)
        else:
            # Match to previous based on continuity
            if len(modes) >= 2:
                # Find best matching
                dist_aa = abs(modes[0] - mode_a[-1])
                dist_ab = abs(modes[0] - mode_b[-1])
                dist_ba = abs(modes[1] - mode_a[-1])
                dist_bb = abs(modes[1] - mode_b[-1])
                
                if dist_aa + dist_bb < dist_ab + dist_ba:
                    mode_a.append(modes[0])
                    mode_b.append(modes[1])
                else:
                    mode_a.append(modes[1])
                    mode_b.append(modes[0])
            elif len(modes) == 1:
                # Only one mode found, match to closest
                if abs(modes[0] - mode_a[-1]) < abs(modes[0] - mode_b[-1]):
                    mode_a.append(modes[0])
                    mode_b.append(mode_b[-1])  # Keep previous
                else:
                    mode_b.append(modes[0])
                    mode_a.append(mode_a[-1])  # Keep previous
    
    mode_a = np.array(mode_a)
    mode_b = np.array(mode_b)
    
    # Calculate properties
    results = []
    for  eigs in [mode_a, mode_b]:
        freq = np.abs(eigs.imag)
        damping = -eigs.real / np.abs(eigs)
        
        results.append({
            'frequency': freq,
            'damping': damping,
            'eigenvalues': eigs
        })
    
    return results

def simulate(properties: WingsuitProperties, initial_y_velocity: float, initial_theta_velocity: float, airspeed: float):
    y0 = [0, 0, initial_theta_velocity, initial_y_velocity]
    
    t_span = (0, 1)
    t_eval = np.linspace(*t_span, 100)

    A = get_coef_matrix(properties, airspeed)

    return solve_ivp(lambda t, Y: A @ Y, t_span, y0, method='BDF',
                t_eval=t_eval,  # This forces output at these times
                rtol=1e-6, 
                atol=1e-9,
                max_step=0.001)

