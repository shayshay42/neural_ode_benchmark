import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np

class ODEFunc(nn.Module):
    def __init__(self, input_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.Tanh(),
            nn.Linear(50, input_dim)
        )
        
    def forward(self, t, y):
        return self.net(y)

class NeuralODE(nn.Module):
    def __init__(self, func):
        super(NeuralODE, self).__init__()
        self.func = func

    def forward(self, y0, t):
        return odeint(self.func, y0, t)

def train_neural_ode(model, data, times, num_epochs, device, patience=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)  # Increased learning rate
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(data.shape[0]):
            optimizer.zero_grad()
            pred = model(torch.Tensor(data[i, 0]).to(device), torch.Tensor(times).to(device))
            loss = torch.mean((pred - torch.Tensor(data[i]).to(device)) ** 2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if total_loss < best_loss:
            best_loss = total_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}')

    return model

def test_neural_ode(model, true_func, test_initial_conditions, t, params, device):
    mse_list = []
    r2_list = []

    for initial_condition in test_initial_conditions:
        with torch.no_grad():
            pred = model(torch.Tensor(initial_condition).to(device), torch.Tensor(t).to(device))
            pred = pred.cpu().numpy()
        
        # Unpack params before passing to true_func
        true_solution = true_func(initial_condition, t, *params)
        
        mse = np.mean((true_solution - pred)**2)
        mse_list.append(mse)

        ss_tot = np.sum((true_solution - np.mean(true_solution))**2)
        ss_res = np.sum((true_solution - pred)**2)
        r2 = 1 - (ss_res / ss_tot)
        r2_list.append(r2)

    return np.mean(mse_list), np.mean(r2_list)