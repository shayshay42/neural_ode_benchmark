#!/bin/bash

NUM_SIMULATIONS=30
NUM_EPOCHS=10

echo "System,Training Time,MSE,R-squared" > julia_results.csv

echo "Running Lotka-Volterra benchmark..."
julia --project=node_bench lotka_volterra_benchmark.jl $NUM_SIMULATIONS $NUM_EPOCHS | awk '{print "Lotka-Volterra," $0}' >> julia_results.csv

echo "Running Van der Pol benchmark..."
julia --project=node_bench van_der_pol_benchmark.jl $NUM_SIMULATIONS $NUM_EPOCHS | awk '{print "Van der Pol," $0}' >> julia_results.csv

echo "Running Lorenz benchmark..."
julia --project=node_bench lorenz_benchmark.jl $NUM_SIMULATIONS $NUM_EPOCHS | awk '{print "Lorenz," $0}' >> julia_results.csv

echo "Julia benchmarks completed. Results saved in julia_results.csv"