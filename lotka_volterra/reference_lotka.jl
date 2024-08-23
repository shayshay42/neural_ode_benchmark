using OrdinaryDiffEq
using Plots
using Flux, DiffEqFlux, Optim
using Optimization, OptimizationOptimJL, OptimizationOptimisers


function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end

u0 = Float32[1.0,1.0]

tspan = Float32.((0.0,10.0))
p1 = Float32[1.5,1.0,3.0,1.0]
datasize = 100
t = range(tspan[1], tspan[2], length=datasize)

prob = ODEProblem(lotka_volterra,u0,tspan,p1)
sol = solve(prob,Tsit5())
test_data = Array(solve(prob,Tsit5(),saveat=t))
plot(sol)

tshort = 3.5f0

dudt = Flux.Chain(Flux.Dense(2, 32, tanh),
                  Flux.Dense(32, 2))
p, re = Flux.destructure(dudt)
dudt2_(u,p,t) = re(p)(u)
prob = ODEProblem(dudt2_,u0,(0f0,tshort),p)

function loss(p) # Our 1-layer neural network
  _prob = remake(prob,p=p)
  pred = Array(solve(_prob,Tsit5(),saveat=t[t .<= tshort]))
  sum(abs2, pred - test_data[:,1:size(pred,2)]),pred
end

iter = 0

cb = function (p,l,pred) #callback function to observe training
  global iter += 1
  if iter % 10 == 0
    @show l
    _t = t[t .<= tshort]
    pl = plot(_t,test_data[:,1:size(pred,2)]',markersize=2, label=["true x" "true y"])
    display(scatter!(pl, _t, pred',markersize=2, label=["pred x" "pred y"]))
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

prob = ODEProblem(dudt2_,u0,(0f0,tshort),nothing)

function loss(p) # Our 1-layer neural network
  _prob = remake(prob,p=p)
  pred = Array(solve(_prob,Tsit5(),saveat=t[t .<= tshort]))
  sum(abs2, pred - test_data[:,1:size(pred,2)]),pred
end

iter = 0

cb = function (p,l,pred) #callback function to observe training
  global iter += 1
  if iter % 10 == 0
    @show l
    _t = t[t .<= tshort]
    pl = plot(_t,test_data[:,1:size(pred,2)]',markersize=5, label=["true x" "true y"])
    display(scatter!(pl, _t, pred',markersize=5, label=["pred x" "pred y"]))
  end
  false
end

iter = -1

# or train the initial condition and neural network
res3 = DiffEqFlux.sciml_train(loss, res2.minimizer, BFGS(initial_stepnorm=0.001), maxiters = 1000, allow_f_increases=true, cb = cb)

pl = plot(t,test_data',markersize=5, label=["true x" "true y"])
display(scatter!(pl, t, loss(res3.minimizer)[2]',markersize=5, label=["pred x" "pred y"]))
savefig("lotka_volterra_node.png")