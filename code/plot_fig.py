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
        ntraining = int(match.group(1)) if match else None
        rotation = int(match.group(2)) if match else None
        
        dropout_match = re.search(r'drop_(\d+\.\d+)', filename)
        dropout = float(dropout_match.group(1)) if dropout_match else None
        
        l2_match = re.search(r'L2_(\d+\.\d+)', filename)
        l2_reg = float(l2_match.group(1)) if l2_match else None
        
        if ntraining is not None and rotation is not None:
            results.append({
                "Ntraining": ntraining,
                "Rotation": rotation,
                "Dropout": dropout,
                "L2_reg": l2_reg,
                "FVAF_train": data.get("predict_training_eval", [None, None])[1], 
                "FVAF_val": data.get("predict_validation_eval", [None, None])[1],
                "FVAF_test": data.get("predict_testing_eval", [None, None])[1]
            })
    
    return pd.DataFrame(results).sort_values(["Ntraining", "Rotation"])

# Function to separate data by any column value
def separate_by_column(df, column_name):
    unique_column_values = df[column_name].dropna().unique()
    df_separate = [df[df[column_name] == val] for val in unique_column_values]
    df_mean = [df.groupby("Ntraining").mean() for df in df_separate]
    ntraining_values = df_mean[0].index.tolist()

    return df, ntraining_values, unique_column_values

# Function to plot a figure
def plot_figure(x_values, y_values, labels, title, ylabel, filename):
    plt.figure(figsize=(10, 5))
    for y, label in zip(y_values, labels):
        plt.plot(x_values, y, marker='o', linestyle='-', label=label)
    
    plt.xlabel("Number of Training Folds")
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)
    plt.grid()
    plt.savefig(filename)
    plt.show()

# Figure 1: No Regularization vs. Early Stopping
def plot_figure_1(df):
    df_list, ntraining_values, _ = separate_by_column(df, "Early_Stopping")
    
    plot_figure(ntraining_values, 
                df_list,
                ["No Regularization", "Early Stopping"],
                "Figure 1: Test Set FVAF vs. Training Set Size",
                "FVAF",
                "figure_1.png")

# Figure 2: Dropout Experiments
def plot_figure_2(df_mean):
    df_list, ntraining_values, dropout_values = separate_by_column(df_mean, "Dropout")
    dropout_labels = [f"Dropout: {val}" for val in dropout_values]
    
    plot_figure(ntraining_values, 
                df_list,  # Need separate dropout values here
                dropout_labels,
                "Figure 2: Validation FVAF vs. Training Set Size",
                "FVAF",
                "figure_2.png")

# Figure 3: Dropout + Early Stopping
def plot_figure_3(df_mean):
    df_list, ntraining_values, dropout_values = separate_by_column(df_mean, "Dropout")
    dropout_labels = [f"Early Stopping + Dropout of {val}" for val in dropout_values]

    plot_figure(ntraining_values, 
                df_list,  # Need separate dropout values w/ early stopping
                dropout_labels,
                "Figure 3: Validation FVAF vs. Training Set Size (Dropout + Early Stopping)",
                "FVAF",
                "figure_3.png")

# Figure 4: L2 Regularization
def plot_figure_4(df_mean):
    df_list, ntraining_values, l2_values = separate_by_column(df_mean, "L2_reg")
    l2_labels = [f"L2: {val}" for val in l2_values]

    
    plot_figure(ntraining_values, 
                df_list,
                l2_labels,
                "Figure 4: Validation FVAF vs. Training Set Size (L2 Regularization)",
                "FVAF",
                "figure_4.png")

# Figure 5: Comparing All Methods
def plot_figure_5(df_mean):

    ntraining_values = df_mean[0].index.tolist()
    
    plot_figure(ntraining_values, 
                [df_mean["FVAF_test"]],  # Need all model types compared
                ["All Regularization Techniques"],
                "Figure 5: Test Set FVAF vs. Training Set Size (All Methods)",
                "FVAF",
                "figure_5.png")

# Generate all figures
def generate_all_figures():
    df = load_results()
    
    plot_figure_1(df)
    plot_figure_2(df)
    plot_figure_3(df)
    plot_figure_4(df)
    plot_figure_5(df)

if __name__ == "__main__":
    generate_all_figures()
