using DifferentialEquations
using Flux
using Plots
using Random
using Statistics
using OrdinaryDiffEq
using DiffEqFlux
using Optim

# 1. Lotka-Volterra ODE system
function lotka_volterra(u, p, t)
    x, y = u
    α, β, δ, γ = p
    [α * x - β * x * y, δ * x * y - γ * y]
end

# 2. Simulate ODE for different initial conditions
function simulate_lotka_volterra(initial_conditions, t, params)
    solutions = []
    for ic in eachrow(initial_conditions)
        prob = ODEProblem(lotka_volterra, ic, extrema(t), params)
        sol = solve(prob, Tsit5(), saveat=t)
        push!(solutions, Array(sol))
    end
    return cat(solutions..., dims=3)
end

# Generate data
Random.seed!(42)
num_simulations = 50
t = range(0, 20, length=200)
params = (1.5, 1.0, 3.0, 1.0)  # α, β, δ, γ

initial_conditions = rand(num_simulations, 2) * 10
dataset = simulate_lotka_volterra(initial_conditions, t, params)

# 3. Neural ODE
dudt = FastChain(FastDense(2, 32, tanh),
                 FastDense(32, 2))
p = initial_params(dudt)

function dudt_(u, p, t)
    dudt(u, p)
end

# 4. Training loop
function train_neural_ode(p, data, times, num_epochs)
    function loss(p)
        pred = Array(solve(ODEProblem(dudt_, data[:, 1, 1], extrema(times), p), Tsit5(), saveat=times))
        sum(abs2, pred .- data[:, :, 1]), pred
    end

    iter = 0
    cb = function (p, l, pred)
        global iter += 1
        if iter % 10 == 0
            println("Iteration: $iter, Loss: $l")
            pl = plot(times, data[:, :, 1]', label=["True x" "True y"], linewidth=2)
            display(plot!(pl, times, pred', label=["Predicted x" "Predicted y"], linestyle=:dash, linewidth=2))
        end
        false
    end

    res1 = DiffEqFlux.sciml_train(loss, p, ADAM(0.01), maxiters=num_epochs, cb=cb)
    res2 = DiffEqFlux.sciml_train(loss, res1.minimizer, BFGS(initial_stepnorm=0.01), maxiters=100, allow_f_increases=true, cb=cb)
    
    return res2.minimizer
end

# 5. Test script
function test_neural_ode(p, true_func, initial_condition, t)
    pred = Array(solve(ODEProblem(dudt_, initial_condition, extrema(t), p), Tsit5(), saveat=t))
    
    prob = ODEProblem(true_func, initial_condition, extrema(t), params)
    true_solution = Array(solve(prob, Tsit5(), saveat=t))
    
    plot(t, true_solution[1, :], label="True x", color=:blue, linewidth=2)
    plot!(t, true_solution[2, :], label="True y", color=:red, linewidth=2)
    plot!(t, pred[1, :], label="Predicted x", color=:blue, linestyle=:dash, linewidth=2)
    plot!(t, pred[2, :], label="Predicted y", color=:red, linestyle=:dash, linewidth=2)
    plot!(legend=:outerbottom, legendcolumns=4)
    title!("Neural ODE vs True Lotka-Volterra")
    xlabel!("Time")
    ylabel!("Population")
    savefig("lotka_volterra_comparison.png")
    display(current())
end

# Main execution
trained_params = train_neural_ode(p, dataset, t, 1000)

test_initial_condition = [5.0, 5.0]
test_neural_ode(trained_params, lotka_volterra, test_initial_condition, collect(t))