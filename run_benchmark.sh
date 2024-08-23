#!/bin/bash

echo "Starting benchmarks..."

echo "Running Julia benchmarks..."
bash run_julia_benchmark.sh

echo "Running PyTorch benchmarks..."
python benchmark.py

echo "Plotting results..."
python plot_results.py

echo "Benchmark completed. Results plotted in benchmark_results.png"