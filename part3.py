import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=200)

k = 1500
k_t = 200
d = 150
d_t = 0.03
l_t = 0.1
l_a = 0.05

m = 2
I = 0.13

C_t = 0.4
C_y = 0.6

# Part b numerical integration

plt.figure()

def system(t, Y, U):
    y, y_dot = Y
    
    return [
        y_dot, # Indicate that y_dot is the derivative of y
        (C_y * U - d) * y_dot / m - (k / m) * y
    ]

U_values = [240, 260]

for U in U_values:
    y0 = [0, 5]
    
    t_span = (0, 1)
    t_eval = np.linspace(*t_span, 100)

    solution = solve_ivp(system, t_span, y0, t_eval=t_eval, args=(U,))
    plt.plot(solution.t, solution.y[0], label=f'y(t) when U = {U}')

plt.xlabel('t')
plt.ylabel('y')
plt.title("Solutions on either side of critical airspeed")
plt.legend()
plt.show(block=False)

# Part C modal analysis

# State structure is:
# _____________
# | theta     |
# | y         |
# | theta_dot |
# | y_dot     |
# _____________

def get_coef_matrix(U):
    theta_ddot_coef = I - m * l_t * l_t

    theta_ddot_rhs = np.array([
        C_t * U * U + l_a * C_t * U * U - k_t,
        -l_t * k,
        -d_t,
        C_y * l_a * U + l_t * (C_y * U - d)
    ]) / theta_ddot_coef

    y_ddot_coef = 1 - l_t * l_t * m / I

    y_ddot_rhs = np.array([
        l_t * (l_a * C_t * U * U - k_t) / I, # theta coef
        -k / m, # y coef
        C_t * U * U / m - d_t * l_t / I, # theta_dot coef
        C_y * l_a * l_t * U / I + (C_y * U - d) / m, # y_dot coef
    ]) / y_ddot_coef

    A = np.row_stack([
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        theta_ddot_rhs,
        y_ddot_rhs
    ])

    return A


U_vals = np.linspace(0, 500, 501)

eig_sets = [np.linalg.eig(get_coef_matrix(U))[0] for U in U_vals]

for eig_set, U in zip(eig_sets, U_vals):
    print(f"{U}: {eig_set}")
    # [eig for eig in eig_set if eig.imag > 0][0]

eigs = np.array([max(eig_set, key=lambda eig: eig.imag) for eig_set in eig_sets])

damped_frequency = abs(eigs.imag)

damping_ratio = -eigs.real / damped_frequency

fig, axes = plt.subplots(1, 2)

axes[0].plot(U_vals, damped_frequency)
axes[1].plot(U_vals, damping_ratio)

print(np.linalg.eig(get_coef_matrix(200))[0])

plt.show()

