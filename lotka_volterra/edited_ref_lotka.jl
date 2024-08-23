using OrdinaryDiffEq
using Plots
using Flux, DiffEqFlux, Optim
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using CUDA
using GalacticOptim

# Check if CUDA is available
if CUDA.functional()
    @info "CUDA is available, using GPU"
    CUDA.allowscalar(false)
else
    @warn "CUDA is not available, falling back to CPU"
end

function lotka_volterra!(du,u,p,t)
    x, y = u
    α, β, δ, γ = p
    du[1] = α*x - β*x*y
    du[2] = -δ*y + γ*x*y
end

# Use CuArray if CUDA is available, otherwise use regular Array
array_type = CUDA.functional() ? CuArray : Array

u0 = array_type(Float32[1.0,1.0])
tspan = (0.0f0, 10.0f0)
p1 = array_type(Float32[1.5,1.0,3.0,1.0])
datasize = 100
t = array_type(range(tspan[1], tspan[2], length=datasize))

prob = ODEProblem(lotka_volterra!,u0,tspan,p1)
sol = solve(prob,Tsit5())
test_data = array_type(Array(solve(prob,Tsit5(),saveat=t)))

tshort = 3.5f0

dudt = Flux.Chain(Flux.Dense(2, 32, tanh),
                  Flux.Dense(32, 2)) |> gpu

p, re = Flux.destructure(dudt)
p = array_type(p)

dudt2_(u,p,t) = re(p)(u)
prob_node = ODEProblem(dudt2_,u0,(0f0,tshort),p)

function loss(p)
    _prob = remake(prob_node,p=p)
    pred = Array(solve(_prob,Tsit5(),saveat=t[t .<= tshort], sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP())))
    sum(abs2, pred - Array(test_data[:,1:size(pred,2)])), pred
end

iter = 0

cb = function (p,l,pred)
    global iter += 1
    if iter % 10 == 0
        @show l
        _t = Array(t[t .<= tshort])
        pl = plot(_t, Array(test_data[:,1:size(pred,2)])', markersize=2, label=["true x" "true y"])
        display(scatter!(pl, _t, pred', markersize=2, label=["pred x" "pred y"]))
    end
    false
end

iter = -1

# Train the initial condition and neural network
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((p,_)->loss(p)[1], adtype)
optprob = Optimization.OptimizationProblem(optf, p)

res1 = Optimization.solve(optprob, OptimizationOptimisers.ADAM(0.01), callback=cb, maxiters=1000)
res2 = Optimization.solve(optprob, Optim.BFGS(initial_stepnorm=0.01), callback=cb, maxiters=1000)

tshort = 10f0

prob_node = ODEProblem(dudt2_,u0,(0f0,tshort),res2.u)

function loss(p)
    _prob = remake(prob_node,p=p)
    pred = Array(solve(_prob,Tsit5(),saveat=t[t .<= tshort], sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP())))
    sum(abs2, pred - Array(test_data[:,1:size(pred,2)])), pred
end

iter = 0

cb = function (p,l,pred)
    global iter += 1
    if iter % 10 == 0
        @show l
        _t = Array(t[t .<= tshort])
        pl = plot(_t, Array(test_data[:,1:size(pred,2)])', markersize=5, label=["true x" "true y"])
        display(scatter!(pl, _t, pred', markersize=5, label=["pred x" "pred y"]))
    end
    false
end

iter = -1

# Train the initial condition and neural network
optf = Optimization.OptimizationFunction((p,_)->loss(p)[1], adtype)
optprob = Optimization.OptimizationProblem(optf, res2.u)
res3 = Optimization.solve(optprob, Optim.BFGS(initial_stepnorm=0.001), callback=cb, maxiters=1000)

pl = plot(Array(t), Array(test_data)', markersize=5, label=["true x" "true y"])
display(scatter!(pl, Array(t), Array(loss(res3.u)[2])', markersize=5, label=["pred x" "pred y"]))
savefig("lotka_volterra_node.png")