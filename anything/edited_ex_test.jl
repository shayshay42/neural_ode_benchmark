using Lux, DiffEqFlux, OrdinaryDiffEq, Optimization, OptimizationOptimJL,
      OptimizationOptimisers, Random, Plots, ComponentArrays

rng = Random.default_rng()
Random.seed!(rng, 0)

u0 = Float32[2.0; 0.0]
datasize = 30
tspan = (0.0f0, 1.5f0)
tsteps = range(tspan[1], tspan[2], length = datasize)

function trueODEfunc(du, u, p, t)
    true_A = [-0.1 2.0; -2.0 -0.1]
    du .= ((u .^ 3)'true_A)'
end

prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

dudt2 = Lux.Chain(x -> x .^ 3,
                  Lux.Dense(2, 50, tanh),
                  Lux.Dense(50, 2))
p, st = Lux.setup(rng, dudt2)

# Convert the parameters to a flat vector
function flatten_params(p)
    return vcat([vec(x) for x in Iterators.flatten(values(p))]...)
end

p_flat = flatten_params(p)

function dudt(u, p, t)
    x = u
    ps = Lux.unpack(dudt2, p)
    for layer in dudt2
        x = layer(x, ps[layer], st[layer])[1]
    end
    return x
end

prob_neuralode = NeuralODE(dudt, tspan, Tsit5(), saveat = tsteps)

function predict_neuralode(p)
    Array(prob_neuralode(u0, p, st)[1])
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, ode_data .- pred)
    return loss, pred
end

callback = function (p, l, pred; doplot = false)
    println("Loss: ", l)
    if doplot
        plt = scatter(tsteps, ode_data[1, :], label = "data")
        scatter!(plt, tsteps, pred[1, :], label = "prediction")
        display(plot(plt))
    end
    return false
end

# Use p_flat directly
callback(p_flat, loss_neuralode(p_flat)..., doplot = true)

# use Optimization.jl to solve the problem
adtype = Optimization.AutoZygote()

optf = Optimization.OptimizationFunction((x, p) -> loss_neuralode(x)[1], adtype)
optprob = Optimization.OptimizationProblem(optf, p_flat)

result_neuralode = Optimization.solve(optprob,
    OptimizationOptimisers.Adam(0.05),
    callback = callback,
    maxiters = 300)

optprob2 = remake(optprob, u0 = result_neuralode.u)

result_neuralode2 = Optimization.solve(optprob2,
    Optim.BFGS(initial_stepnorm = 0.01),
    callback = callback,
    allow_f_increases = false)

callback(result_neuralode2.u, loss_neuralode(result_neuralode2.u)..., doplot = true)