import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

def load_results_fig2():
    """
    Load results from multiple experiments.

    :param directory: Directory containing result files
    :param ntraining_values: List of training set sizes used in experiments
    :return: DataFrame with results
    """
    results = []

    files = [f for f in os.listdir("results/") if f.startswith("bmi_") and f.endswith(".pkl")]

    for filename in files:

        filename = "results/" + filename
        print(f"load fname: '{filename}'")
        
        if os.path.exists(filename):
            with open(filename, "rb") as fp:
                data = pickle.load(fp)

            match = re.search(r'Ntraining_(\d+)_', filename)
            n = int(match.group(1)) if match else None

            if n is not None:
                results.append({
                    "Ntraining": n,
                    "FVAF_train": data["predict_training_eval"][1], 
                    "RMSE_train": data["predict_training_eval"][2],  
                    "FVAF_val": data["predict_validation_eval"][1],  
                    "RMSE_val": data["predict_validation_eval"][2],  
                    "FVAF_test": data["predict_testing_eval"][1],  
                    "RMSE_test": data["predict_testing_eval"][2]
                })
            else:
                print("Cannot parse Ntraining number in reading pkl fname to plot fig2")

    if len(results) == 0:
        print("No .pkl files found")
        return None

    return pd.DataFrame(results).sort_values("Ntraining")

def plot_figure_2():
    """
    Generate Figure 2: FVAF and RMSE vs. Training Set Size using Matplotlib.

    :param ntraining_values: List of training set sizes used in experiments
    """
    
    # Loading results
    results_df = load_results_fig2()

    if results_df is None:
        print("No results found");
        return None;

    # Plot figure 2a
    
    plt.figure(figsize=(10, 5))

    plt.plot(results_df["Ntraining"], results_df["FVAF_train"], 'bo-', label="FVAF (Train)")
    plt.plot(results_df["Ntraining"], results_df["FVAF_val"], 'go-', label="FVAF (Validation)")
    plt.plot(results_df["Ntraining"], results_df["FVAF_test"], 'ro-', label="FVAF (Test)")
    plt.ylabel("FVAF")
    plt.xlabel("Number of Training Folds")
    plt.legend()
    plt.title("Figure 2a: FVAF vs. Training Set Size")

    plt.savefig("figure_2a.png")

    # Plot figure 2b
    
    plt.figure(figsize=(10, 5))

    plt.plot(results_df["Ntraining"], results_df["RMSE_train"], 'bs--', label="RMSE (Train)")
    plt.plot(results_df["Ntraining"], results_df["RMSE_val"], 'gs--', label="RMSE (Validation)")
    plt.plot(results_df["Ntraining"], results_df["RMSE_test"], 'rs--', label="RMSE (Test)")
    plt.ylabel("RMSE")
    plt.xlabel("Number of Training Folds")
    plt.legend()
    plt.title("Figure 2b: RMSE vs. Training Set Size")

    plt.savefig("figure_2b.png")

    # Log to wandb
    wandb.log({"Figure 2a": wandb.Image("figure_2a.png"), "Figure 2b": wandb.Image("figure_2b.png")})

def plot_figure_1(time_testing, outs_testing, predict_testing):
    """
    Generates Figure 1: True Acceleration vs. Predicted Velocity (Shoulder & Elbow) using Matplotlib.

    :param time_testing: (1498, 1) timestamps for the test fold.
    :param outs_testing: (1498, 2) true acceleration (shoulder, elbow).
    :param predict_testing: (1498, 2) predicted velocity (shoulder, elbow).
    """

    # Ensure time is a 1D array
    time_testing = np.array(time_testing).flatten()

    # Extract shoulder and elbow data separately
    true_accel_shoulder = outs_testing[:, 0]  # First column
    pred_vel_shoulder = predict_testing[:, 0]  # First column

    # Stretch the x axis
    plt.figure(figsize=(30, 5))
    
    # Plot true acceleration
    plt.plot(time_testing, true_accel_shoulder, label="True Acceleration", linestyle="dashed")

    # Plot predicted velocity
    plt.plot(time_testing, pred_vel_shoulder, label="Predicted Velocity", linestyle="solid")

    # Formatting
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration / Velocity")
    plt.title("True Acceleration vs Predicted Velocity Over Time")
    plt.legend()
    plt.grid()

    # Save and log to wandb
    plt.savefig("figure_1.png")
    wandb.log({"Figure 1": wandb.Image("figure_1.png")})
