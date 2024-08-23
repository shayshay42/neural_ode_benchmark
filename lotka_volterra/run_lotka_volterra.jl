using OrdinaryDiffEq
using Plots
using Flux, DiffEqFlux, Optim

function lotka_volterra(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = α*x - β*x*y
    du[2] = -δ*y + γ*x*y
end

function run_benchmark(num_simulations, t_span, params, num_epochs)
    u0 = Float32[1.0, 1.0]
    tspan = Float32.(t_span)
    datasize = 100
    t = range(tspan[1], tspan[2], length=datasize)

    prob = ODEProblem(lotka_volterra, u0, tspan, params)
    sol = solve(prob, Tsit5())
    test_data = Array(solve(prob, Tsit5(), saveat=t))

    plot(sol)
    savefig("true_solution.png")
    println("True solution plot saved as true_solution.png")

    tshort = 3.5f0

    dudt = FastChain(FastDense(2, 32, tanh),
                     FastDense(32, 2))
    p = initial_params(dudt)
    dudt2_(u, p, t) = dudt(u, p)
    prob_node = ODEProblem(dudt2_, u0, (0f0, tshort), nothing)

    function loss(p)
        _prob = remake(prob_node, p=p)
        pred = Array(solve(_prob, Tsit5(), saveat=t[t .<= tshort]))
        sum(abs2, pred - test_data[:, 1:size(pred, 2)]), pred
    end

    iter = 0
    cb = function (p, l, pred)
        global iter += 1
        if iter % 10 == 0
            @show l
            _t = t[t .<= tshort]
            pl = plot(_t, test_data[:, 1:size(pred, 2)]', markersize=2, label=["true x" "true y"])
            display(scatter!(pl, _t, pred', markersize=2, label=["pred x" "pred y"]))
        end
        false
    end

    # Training
    println("Starting training...")
    res1 = DiffEqFlux.sciml_train(loss, p, ADAM(0.01), maxiters=num_epochs, cb=cb)
    res2 = DiffEqFlux.sciml_train(loss, res1.minimizer, BFGS(initial_stepnorm=0.01), maxiters=num_epochs, allow_f_increases=true, cb=cb)

    # Final prediction
    tshort = 10f0
    prob_node = ODEProblem(dudt2_, u0, (0f0, tshort), nothing)
    
    final_loss, final_pred = loss(res2.minimizer)

    pl = plot(t, test_data', markersize=5, label=["true x" "true y"])
    scatter!(pl, t, final_pred', markersize=5, label=["pred x" "pred y"])
    savefig(pl, "lotka_volterra_node.png")
    println("Final prediction plot saved as lotka_volterra_node.png")

    return final_loss
end

# Set parameters
num_simulations = 1  # Not used in this version, but kept for consistency
t_span = (0.0, 10.0)
params = Float32[1.5, 1.0, 3.0, 1.0]
num_epochs = 1000

# Run benchmark
final_loss = run_benchmark(num_simulations, t_span, params, num_epochs)
println("Final loss: $final_loss")