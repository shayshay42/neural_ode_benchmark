import pandas as pd
import matplotlib.pyplot as plt
import json

def load_julia_results():
    return pd.read_csv("julia_results.csv")

def load_pytorch_results():
    with open("pytorch_results.json", "r") as f:
        data = json.load(f)
    
    rows = []
    for item in data:
        rows.append({
            "System": item["system"],
            "Training Time": item["pytorch"]["time"],
            "MSE": item["pytorch"]["mse"],
            "R-squared": item["pytorch"]["r2"]
        })
    
    return pd.DataFrame(rows)

def plot_results(julia_df, pytorch_df):
    systems = julia_df["System"].unique()
    metrics = ["Training Time", "MSE", "R-squared"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(metrics):
        julia_values = julia_df[metric].values
        pytorch_values = pytorch_df[metric].values
        
        x = range(len(systems))
        width = 0.35
        
        axes[i].bar([xi - width/2 for xi in x], julia_values, width, label="Julia")
        axes[i].bar([xi + width/2 for xi in x], pytorch_values, width, label="PyTorch")
        
        axes[i].set_ylabel(metric)
        axes[i].set_title(f"{metric} Comparison")
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(systems, rotation=45)
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig("benchmark_results.png")
    plt.show()

def main():
    julia_df = load_julia_results()
    pytorch_df = load_pytorch_results()
    
    print("Julia Results:")
    print(julia_df)
    print("\nPyTorch Results:")
    print(pytorch_df)
    
    plot_results(julia_df, pytorch_df)

if __name__ == "__main__":
    main()