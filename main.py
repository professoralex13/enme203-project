import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider 

from wingsuit import WingsuitProperties, get_wingsuit_eigs, simulate

MAX_SPEED = 550 / 3.6 # Converts kmh to m/s

np.set_printoptions(linewidth=200)

default_properties: WingsuitProperties = {
    'k': 1500,
    'k_t': 200,

    'd': 150,
    'd_t': 0.03,
    
    'l_t': 0.1,
    'l_a': 0.05,

    'm': 2,
    'I': 0.13,

    'C_t': 0.4,
    'C_y': 0.6
}

# Part b numerical integration

def part_b():
    fig = plt.figure()
    axis = plt.axes()

    U_values = [240, 260]

    for U in U_values:
        new_properties = default_properties.copy()
        new_properties['l_t'] = 0
        new_properties['C_t'] = 0

        solution = simulate(new_properties, 1, 1, U)

        axis.plot(solution.t, solution.y[1], label=f'y(t) when U = {U}')

    axis.set_xlabel('t')
    axis.set_ylabel('y')
    axis.set_title("Solutions on either side of critical airspeed")
    fig.legend()

part_b()

# Part C modal analysis

def plot_eigs(U_vals, eigs, axes, include_decoupled = False):
    real_axis, imaginary_axis = axes

    real_axis.set_xlabel('Airspeed (m/s)')
    real_axis.set_title(f"Real")
    real_axis.set_ylabel(f"Real Eigenvector Components")

    real_axis.invert_yaxis()
    real_axis.axhline(y=0)

    imaginary_axis.set_xlabel('Airspeed (m/s)')
    imaginary_axis.set_title('Imaginary')
    imaginary_axis.set_ylabel(f"Imaginary Eigenvector Component")

    frequency_decoupled = np.sqrt((default_properties['k_t'] - default_properties['l_a'] * default_properties['C_t'] * U_vals * U_vals) / default_properties['I'])
    damping_ratio_decoupled = (default_properties['d'] - default_properties['C_y'] * U_vals) / np.sqrt(default_properties['k'] * default_properties['m'])

    lines = { 'real': [], 'imaginary': [] }

    for i, eig_set in enumerate(eigs):
        [real_line] = real_axis.plot(U_vals, eig_set.real, label=f"Eigenvalue {i + 1}")
        [imaginary_line] = imaginary_axis.plot(U_vals, eig_set.imag, label=f"Eigenvalue {i + 1}")

        lines['real'].append(real_line)
        lines['imaginary'].append(imaginary_line)

    if include_decoupled:
        imaginary_axis.plot(U_vals, frequency_decoupled, 'b.', label="Decoupled Frequency")

    real_axis.legend(loc="upper left")
    imaginary_axis.legend()

    return lines

def part_c():
    U_vals = np.arange(0, MAX_SPEED, 1)

    eigs = get_wingsuit_eigs(default_properties, U_vals)

    fig, axes = plt.subplots(1, 2)

    plot_eigs(U_vals, eigs, axes, True)

part_c()

# Part d design modification

def part_d():
    fig, axes = plt.subplots(1, 3)

    U_vals = np.arange(0, MAX_SPEED, 1)
    
    target_U = 330 / 3.6

    new_properties = default_properties.copy()

    new_properties['l_t'] *= -0.5
    new_properties['l_a'] *= 0.5

    new_solution = simulate(new_properties, 1, 1, target_U)
    new_eigs = get_wingsuit_eigs(new_properties, U_vals)

    eig_lines = plot_eigs(U_vals, new_eigs, axes[1:])

    [response_line] = axes[0].plot(new_solution.t, new_solution.y[1])
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('y')
    axes[0].set_title(f'New y(t) when U = {target_U} m/s')

    axes[1].axvline(x=target_U)
    
part_d()

plt.ion()
plt.show(block=True)

