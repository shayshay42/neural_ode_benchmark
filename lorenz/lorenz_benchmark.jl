include("lorenz.jl")
include("neural_ode_julia.jl")

function run_benchmark(num_simulations, t_span, params, num_epochs)
    dataset, t = generate_dataset(num_simulations, t_span, params)
    
    start_time = time()
    
    model = NeuralODE(ODEFunc())
    
    trained_params = train_neural_ode(model, dataset, t, num_epochs)
    
    training_time = time() - start_time
    
    test_initial_conditions = rand(10, 3) .* 30 .- 15
    mse, r2 = test_neural_ode(trained_params, system, test_initial_conditions, t, params)
    
    println("$training_time,$mse,$r2")
end

num_simulations = parse(Int, ARGS[1])
t_span = (0, 10)
params = (10.0, 28.0, 8/3)
num_epochs = parse(Int, ARGS[2])

run_benchmark(num_simulations, t_span, params, num_epochs)