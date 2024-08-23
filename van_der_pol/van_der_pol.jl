using OrdinaryDiffEq

function system(du, u, p, t)
    x, y = u
    μ = p[1]
    du[1] = y
    du[2] = μ * (1 - x^2) * y - x
end

function generate_dataset(num_simulations, t_span, params)
    tspan = (t_span[1], t_span[2])
    t = range(tspan[1], tspan[2], length=200)
    
    initial_conditions = rand(num_simulations, 2) .* 10 .- 5
    dataset = []
    
    for ic in eachrow(initial_conditions)
        prob = ODEProblem(system, ic, tspan, params)
        sol = solve(prob, Tsit5(), saveat=t)
        push!(dataset, Array(sol))
    end
    
    return cat(dataset..., dims=3), collect(t)
end