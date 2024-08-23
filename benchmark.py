import time
import numpy as np
import torch
import json
import multiprocessing as mp
from neural_ode_pytorch import ODEFunc, NeuralODE, train_neural_ode, test_neural_ode
from lotka_volterra import generate_dataset as lv_generate_dataset, system as lv_system
from van_der_pol import generate_dataset as vdp_generate_dataset, system as vdp_system
from lorenz import generate_dataset as lorenz_generate_dataset, system as lorenz_system

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def run_pytorch_benchmark(generate_dataset, system, params, num_simulations, t_span, num_epochs):
    dataset, t = generate_dataset(num_simulations, t_span, params)
    
    start_time = time.time()
    
    input_dim = dataset.shape[2]
    func = ODEFunc(input_dim).to(device)
    model = NeuralODE(func).to(device)
    
    trained_model = train_neural_ode(model, dataset, t, num_epochs, device)
    
    training_time = time.time() - start_time
    
    test_initial_conditions = np.random.rand(10, input_dim) * 10 - 5
    mse, r2 = test_neural_ode(trained_model, system, test_initial_conditions, t, params, device)
    
    return training_time, mse, r2, trained_model

def run_system_benchmark(system_info):
    system_name, generate_dataset, system, params = system_info
    num_simulations = 30
    t_span = (0, 10)
    num_epochs = 10
    
    pytorch_time, pytorch_mse, pytorch_r2, trained_model = run_pytorch_benchmark(generate_dataset, system, params, num_simulations, t_span, num_epochs)
    print(f"{system_name},PyTorch,{pytorch_time:.2f},{pytorch_mse:.6f},{pytorch_r2:.6f}")
    
    # Save PyTorch model weights
    torch.save(trained_model.state_dict(), f"{system_name.lower().replace(' ', '_')}_pytorch_weights.pth")
    
    return {
        "system": system_name,
        "pytorch": {"time": pytorch_time, "mse": pytorch_mse, "r2": pytorch_r2}
    }

def run_benchmark():
    systems = [
        ("Lotka-Volterra", lv_generate_dataset, lv_system, (1.5, 1.0, 3.0, 1.0)),
        ("Van der Pol", vdp_generate_dataset, vdp_system, (1.0,)),
        ("Lorenz", lorenz_generate_dataset, lorenz_system, (10.0, 28.0, 8/3))
    ]
    
    print("System,Framework,Training Time,MSE,R-squared")
    
    with mp.Pool(processes=min(mp.cpu_count(), len(systems))) as pool:
        results = pool.map(run_system_benchmark, systems)
    
    # Save final results
    with open("pytorch_results.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    run_benchmark()