import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from hyper_parameters import *

# Directory containing result files
RESULTS = ["./results/"]

# Function to load experiment results
def load_results(results_dir=RESULTS):
    results = []
    files = []
    for r_dir in results_dir:
        files.extend([os.path.join(r_dir, f) for f in os.listdir(r_dir) if f.startswith("bmi") and f.endswith(".pkl") and ('rotation_0' in f or 'rotation_18' in f)])

    for filename in files:
        with open(filename, "rb") as fp:
            data = pickle.load(fp)

        rotation = data["args"].rotation
        ntraining = data["args"].Ntraining
        dropout = data["args"].dropout
        l2_reg = data["args"].L2_regularization

        # rotation = data.get("rotation")
        # ntraining = data.get("Ntraining")
        # dropout = data.get("dropout")
        # l2_reg = data.get("L2_regularization")

        # early_stopping = data["args"].early_stopping

        if ntraining is not None and rotation is not None:
            results.append({
                "Ntraining": ntraining,
                "Rotation": rotation,
                "Dropout": dropout,
                "L2_reg": l2_reg,
                # "Early_Stopping": early_stopping,
                "FVAF_train": data.get("predict_training_eval", [None, None])[1], 
                "FVAF_val": data.get("predict_validation_eval", [None, None])[1],
                "FVAF_test": data.get("predict_testing_eval", [None, None])[1]
            })
    
    return pd.DataFrame(results).sort_values(["Ntraining", "Rotation"])

# Function to separate data by any column value
def separate_by_column(df, column_name):
    unique_column_values = df[column_name].dropna().unique()
    
    # Create a list of DataFrames, each corresponding to a unique value in column_name
    df_separate = [df[df[column_name] == val] for val in unique_column_values]
    
    # Compute mean for each separated DataFrame
    df_mean = [d.groupby("Ntraining").mean() for d in df_separate]
    
    # Extract training set values
    ntraining_values = df_mean[0].index.tolist() if df_mean else []

    return df_mean, ntraining_values, unique_column_values


# Function to plot a figure
def plot_figure(x_values, y_values, labels, title, ylabel, filename):
    plt.figure(figsize=(10, 5))

    for y, label in zip(y_values, labels):
        plt.plot(x_values, y, marker='o', linestyle='-', label=label)
    
    plt.xlabel("Number of Training Folds")
    plt.ylabel(ylabel)
    # plt.ylim([-1, 1])
    plt.legend()
    plt.title(title)
    plt.grid()
    plt.savefig(filename)
    plt.show()

# Figure 1: No Regularization vs. Early Stopping
def plot_figure_1():
    df = load_results(["./part_1_pkl/", "./results_part2/"])
    df_list, ntraining_values, _ = separate_by_column(df, "Early_Stopping")

    plot_figure(ntraining_values, 
                [df["FVAF_test"] for df in df_list],
                ["No Regularization", "Early Stopping"],
                "Figure 1: Test Set FVAF vs. Training Set Size",
                "FVAF",
                "figure_1.png")

# Figure 2: Dropout Experiments
def plot_figure_2():
    df = load_results(["./results_part3/"])
    
    df_list, ntraining_values, dropout_values = separate_by_column(df, "Dropout")
    dropout_labels = [f"Dropout: {val}" for val in dropout_values]

    print(ntraining_values)
    print(df_list)
    
    plot_figure(ntraining_values, 
                [df["FVAF_val"] for df in df_list],
                dropout_labels,
                "Figure 2: Validation FVAF vs. Training Set Size",
                "FVAF",
                "figure_2.png")

# Figure 3: Dropout + Early Stopping
def plot_figure_3():
    df = load_results(["./results_part4/"])
    
    df_list, ntraining_values, dropout_values = separate_by_column(df, "Dropout")
    dropout_labels = [f"Early Stopping + Dropout of {val}" for val in dropout_values]

    plot_figure(ntraining_values, 
                [df["FVAF_val"] for df in df_list],
                dropout_labels,
                "Figure 3: Validation FVAF vs. Training Set Size (Dropout + Early Stopping)",
                "FVAF",
                "figure_3.png")

# Figure 4: L2 Regularization
def plot_figure_4():
    df = load_results(["./results_part5/"])
    
    df_list, ntraining_values, l2_values = separate_by_column(df, "L2_reg")
    l2_labels = [f"L2: {val}" for val in l2_values]

    
    plot_figure(ntraining_values, 
                [d["FVAF_val"] for d in df_list],
                l2_labels,
                "Figure 4: Validation FVAF vs. Training Set Size (L2 Regularization)",
                "FVAF",
                "figure_4.png")

# Figure 5: Test Performance Across All Model Types    
def plot_figure_5(df):
    """
    Generates Figure 5: Mean test set FVAF as a function of training set size for all five cases.
    
    :param df: DataFrame containing test results
    """
    results, best_dropout, best_l2 = compute_best_test_performance(df)

    plt.figure(figsize=(10, 5))

    for label, values in results.items():
        plt.plot(values.index, values, marker='o', linestyle='-', label=label)

    plt.xlabel("Number of Training Folds")
    plt.ylabel("Mean FVAF (Test Set)")
    plt.legend()
    plt.title("Figure 5: Test Performance Across All Model Types")
    plt.grid()
    plt.savefig("figure_5.png")
    plt.show()

    print(f"Best Dropout: {best_dropout}, Best L2 Regularization: {best_l2}")

# Generate all figures
def generate_all_figures():    
    # plot_figure_1()
    # plot_figure_2()
    # plot_figure_3()
    plot_figure_4()
    # plot_figure_5()

if __name__ == "__main__":
    generate_all_figures()
