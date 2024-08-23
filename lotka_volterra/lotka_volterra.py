import numpy as np
from scipy.integrate import odeint

def system(state, t, alpha, beta, delta, gamma):
    x, y = state
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

def simulate(initial_conditions, t, params):
    alpha, beta, delta, gamma = params
    solutions = []
    for ic in initial_conditions:
        solution = odeint(system, ic, t, args=(alpha, beta, delta, gamma))
        solutions.append(solution)
    return np.array(solutions)

def generate_dataset(num_simulations, t_span, params):
    np.random.seed(42)
    t = np.linspace(t_span[0], t_span[1], 200)
    initial_conditions = np.random.rand(num_simulations, 2) * 10
    dataset = simulate(initial_conditions, t, params)
    return dataset, t