import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


def dinkelbach_power_allocation(P_total, P_max, P_ini):

    # Initial guess for power allocation
    P = P_total / 2
    P_min = 0.0             # Minimum power constraint
    alpha = 1.0             # Coefficient for the objective function
    beta = 2.0              # Exponent for the objective function
    max_iter = 50
    epsilon = 1e-6          # Convergence threshold.

    P1 = []
    for i in range(max_iter):
        # Define the objective function to be minimized
        def objective_function(x):
            return -((x / P_total) ** beta) * alpha

        # Define the constraint
        constraint = ({'type': 'ineq', 'fun': lambda x: P_min - x},
                      {'type': 'ineq', 'fun': lambda x: x - P_max})

        # Solve the optimization problem
        res = minimize(objective_function, P, constraints=constraint, bounds=[(P_min, P_max)])

        # Update power allocation
        P_new = res.x[0] * P_ini

        # Check for convergence
        if abs(P_new - P) < epsilon:
            P1.append(P_ini)
        else:
            P1.append(P_new)

    return abs(int(np.mean(P1)))

