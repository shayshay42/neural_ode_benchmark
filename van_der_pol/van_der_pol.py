import numpy as np
from scipy.integrate import odeint

def system(state, t, mu):
    x, y = state
    dxdt = y
    dydt = mu * (1 - x**2) * y - x
    return [dxdt, dydt]

def simulate(initial_conditions, t, params):
    mu = params[0]
    solutions = []
    for ic in initial_conditions:
        solution = odeint(system, ic, t, args=(mu,))
        solutions.append(solution)
    return np.array(solutions)

def generate_dataset(num_simulations, t_span, params):
    np.random.seed(42)
    t = np.linspace(t_span[0], t_span[1], 200)
    initial_conditions = np.random.rand(num_simulations, 2) * 4 - 2
    dataset = simulate(initial_conditions, t, params)
    return dataset, t