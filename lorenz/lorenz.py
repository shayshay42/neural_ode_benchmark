import numpy as np
from scipy.integrate import odeint

def system(state, t, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def simulate(initial_conditions, t, params):
    sigma, rho, beta = params
    solutions = []
    for ic in initial_conditions:
        solution = odeint(system, ic, t, args=(sigma, rho, beta))
        solutions.append(solution)
    return np.array(solutions)

def generate_dataset(num_simulations, t_span, params):
    np.random.seed(42)
    t = np.linspace(t_span[0], t_span[1], 200)
    initial_conditions = np.random.rand(num_simulations, 3) * 30 - 15
    dataset = simulate(initial_conditions, t, params)
    return dataset, t