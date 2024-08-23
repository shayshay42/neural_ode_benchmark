import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import odeint
from torchdiffeq import odeint as odeint_torch
import matplotlib.pyplot as plt

# 1. Lotka-Volterra ODE system
def lotka_volterra(state, t, alpha, beta, delta, gamma):
    x, y = state
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# 2. Simulate ODE for different initial conditions
def simulate_lotka_volterra(initial_conditions, t, params):
    alpha, beta, delta, gamma = params
    solutions = []
    for ic in initial_conditions:
        solution = odeint(lotka_volterra, ic, t, args=(alpha, beta, delta, gamma))
        solutions.append(solution)
    return np.array(solutions)

# Generate data
np.random.seed(42)
num_simulations = 50
t = np.linspace(0, 20, 200)
params = (1.5, 1.0, 3.0, 1.0)  # alpha, beta, delta, gamma

initial_conditions = np.random.rand(num_simulations, 2) * 10
dataset = simulate_lotka_volterra(initial_conditions, t, params)

# 3. Neural ODE
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 2)
        )
        
    def forward(self, t, y):
        return self.net(y)

class NeuralODE(nn.Module):
    def __init__(self, func):
        super(NeuralODE, self).__init__()
        self.func = func

    def forward(self, y0, t):
        return odeint_torch(self.func, y0, t)

# 4. Training loop
def train_neural_ode(model, data, times, num_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        pred = model(torch.Tensor(data[0, 0]), torch.Tensor(times))
        loss = torch.mean((pred - torch.Tensor(data[0])) ** 2)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            plot_progress(model, data[0], times)

    return model

# 5. Test script
def test_neural_ode(model, true_func, test_initial_conditions, t):
    mse_list = []
    r2_list = []

    plt.figure(figsize=(15, 10))

    for i, initial_condition in enumerate(test_initial_conditions):
        with torch.no_grad():
            pred = model(torch.Tensor(initial_condition), torch.Tensor(t))
            pred = pred.numpy()
        
        true_solution = odeint(true_func, initial_condition, t, args=params)
        
        # Calculate MSE
        mse = np.mean((true_solution - pred)**2)
        mse_list.append(mse)

        # Calculate R-squared
        ss_tot = np.sum((true_solution - np.mean(true_solution))**2)
        ss_res = np.sum((true_solution - pred)**2)
        r2 = 1 - (ss_res / ss_tot)
        r2_list.append(r2)

        plt.subplot(2, 2, i+1)
        plt.plot(t, true_solution[:, 0], 'b-', label='True x')
        plt.plot(t, true_solution[:, 1], 'r-', label='True y')
        plt.plot(t, pred[:, 0], 'b--', label='Predicted x')
        plt.plot(t, pred[:, 1], 'r--', label='Predicted y')
        plt.legend(loc='lower right', ncol=2)
        plt.title(f'Test Case {i+1}: IC = {initial_condition}')
        plt.xlabel('Time')
        plt.ylabel('Population')

    plt.tight_layout()
    plt.savefig('lotka_volterra_test_cases.png')
    plt.show()

    # Print statistical metrics
    print("\nTest Results:")
    for i, (mse, r2) in enumerate(zip(mse_list, r2_list)):
        print(f"Test Case {i+1}:")
        print(f"  MSE: {mse:.6f}")
        print(f"  R-squared: {r2:.6f}")

    print("\nOverall Performance:")
    print(f"Average MSE: {np.mean(mse_list):.6f}")
    print(f"Average R-squared: {np.mean(r2_list):.6f}")

def plot_progress(model, data, times):
    with torch.no_grad():
        pred = model(torch.Tensor(data[0]), torch.Tensor(times))
        pred = pred.numpy()
    
    plt.figure(figsize=(10, 5))
    plt.plot(times, data[:, 0], 'b-', label='True x')
    plt.plot(times, data[:, 1], 'r-', label='True y')
    plt.plot(times, pred[:, 0], 'b--', label='Predicted x')
    plt.plot(times, pred[:, 1], 'r--', label='Predicted y')
    plt.legend(loc='lower right', ncol=2)
    plt.title('Training Progress')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.show()

# Main execution
if __name__ == "__main__":
    func = ODEFunc()
    model = NeuralODE(func)
    
    trained_model = train_neural_ode(model, dataset, t, num_epochs=1000)
    
    # Multiple test cases
    test_initial_conditions = [
        [5.0, 5.0],
        [1.0, 2.0],
        [8.0, 3.0],
        [2.0, 7.0]
    ]
    test_neural_ode(trained_model, lotka_volterra, test_initial_conditions, t)