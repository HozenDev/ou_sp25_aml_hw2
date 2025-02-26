import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Directory containing result files
RESULTS_DIR = "./results/"

# Function to load experiment results
def load_results():
    results = []
    files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("bmi_") and f.endswith(".pkl")]
    
    for filename in files:
        file_path = os.path.join(RESULTS_DIR, filename)
        with open(file_path, "rb") as fp:
            data = pickle.load(fp)

        match = re.search(r'Ntraining_(\d+)_rotation_(\d+)', filename)
        if match:
            ntraining = int(match.group(1))
            rotation = int(match.group(2))
            
            results.append({
                "Ntraining": ntraining,
                "Rotation": rotation,
                "FVAF_train": data["predict_training_eval"][1], 
                "FVAF_val": data["predict_validation_eval"][1],
                "FVAF_test": data["predict_testing_eval"][1]
            })
    
    return pd.DataFrame(results).sort_values(["Ntraining", "Rotation"])

# Function to plot a figure
def plot_figure(x, y_values, labels, title, ylabel, filename):
    plt.figure(figsize=(10, 5))
    for y, label in zip(y_values, labels):
        plt.plot(x, y, marker='o', linestyle='-', label=label)
    
    plt.xlabel("Number of Training Folds")
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    plt.grid()
    plt.savefig(filename)
    plt.show()

# Figure 1: No Regularization vs. Early Stopping
def plot_figure_1(df_mean, ntraining_values):
    plot_figure(ntraining_values, 
                [df_mean["FVAF_test"], df_mean["FVAF_test"]],  # Need to separate no reg vs early stopping
                ["No Regularization", "Early Stopping"],
                "Figure 1: Test Set FVAF vs. Training Set Size",
                "FVAF",
                "figure_1.png")

# Figure 2: Dropout Experiments
def plot_figure_2(df_mean, ntraining_values):
    plot_figure(ntraining_values, 
                [df_mean["FVAF_val"]],  # Need separate dropout values here
                ["Dropout (varying values)"],
                "Figure 2: Validation FVAF vs. Training Set Size",
                "FVAF",
                "figure_2.png")

# Figure 3: Dropout + Early Stopping
def plot_figure_3(df_mean, ntraining_values):
    plot_figure(ntraining_values, 
                [df_mean["FVAF_val"]],  # Need separate dropout values w/ early stopping
                ["Dropout + Early Stopping"],
                "Figure 3: Validation FVAF vs. Training Set Size (Dropout + Early Stopping)",
                "FVAF",
                "figure_3.png")

# Figure 4: L2 Regularization
def plot_figure_4(df_mean, ntraining_values):
    plot_figure(ntraining_values, 
                [df_mean["FVAF_val"]],  # Need separate L2 values
                ["L2 Regularization (varying values)"],
                "Figure 4: Validation FVAF vs. Training Set Size (L2 Regularization)",
                "FVAF",
                "figure_4.png")

# Figure 5: Comparing All Methods
def plot_figure_5(df_mean, ntraining_values):
    plot_figure(ntraining_values, 
                [df_mean["FVAF_test"]],  # Need all model types compared
                ["All Regularization Techniques"],
                "Figure 5: Test Set FVAF vs. Training Set Size (All Methods)",
                "FVAF",
                "figure_5.png")

# Generate all figures
def generate_all_figures():
    df = load_results()
    df_mean = df.groupby("Ntraining").mean()
    ntraining_values = df_mean.index.tolist()
    
    plot_figure_1(df_mean, ntraining_values)
    plot_figure_2(df_mean, ntraining_values)
    plot_figure_3(df_mean, ntraining_values)
    plot_figure_4(df_mean, ntraining_values)
    plot_figure_5(df_mean, ntraining_values)

if __name__ == "__main__":
    generate_all_figures()
