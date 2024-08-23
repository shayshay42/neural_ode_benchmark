using Flux, DiffEqFlux, OrdinaryDiffEq, Statistics

function ODEFunc()
    Flux.Chain(Flux.Dense(2, 50, tanh), Flux.Dense(50, 2))
end

function NeuralODE(model)
    node = DiffEqFlux.NeuralODE(model, (0.0, 10.0), Tsit5(), saveat=0.1)
    return node
end

function train_neural_ode(model, data, t, num_epochs)
    opt = Flux.ADAM(0.01)
    loss_history = []

    function loss_function(p)
        pred = Array(model(data[:, 1, :], p))
        loss = mean(abs2, data .- pred)
        return loss
    end

    for epoch in 1:num_epochs
        Flux.train!(loss_function, Flux.params(model), [(data,)], opt)
        
        current_loss = loss_function(Flux.params(model))
        push!(loss_history, current_loss)
        
        if epoch % 10 == 0
            println("Epoch $epoch: Loss = $current_loss")
        end
    end

    return Flux.params(model), loss_history
end

function test_neural_ode(trained_params, true_func, test_initial_conditions, t, params)
    model = NeuralODE(ODEFunc())
    Flux.loadparams!(model, trained_params)

    mse_list = []
    r2_list = []

    for initial_condition in eachrow(test_initial_conditions)
        pred = Array(model(initial_condition, trained_params))
        
        prob = ODEProblem(true_func, initial_condition, extrema(t), params)
        true_solution = Array(solve(prob, Tsit5(), saveat=t))
        
        mse = mean((true_solution .- pred).^2)
        push!(mse_list, mse)

        ss_tot = sum((true_solution .- mean(true_solution)).^2)
        ss_res = sum((true_solution .- pred).^2)
        r2 = 1 - (ss_res / ss_tot)
        push!(r2_list, r2)
    end

    return mean(mse_list), mean(r2_list)
end