import os
import pickle
import pandas as pd

from scipy.stats import ttest_rel

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
        early_stopping = data["args"].early_stopping

        if ntraining is not None and rotation is not None:
            results.append({
                "Ntraining": ntraining,
                "Rotation": rotation,
                "Dropout": dropout,
                "L2_reg": l2_reg,
                "Early_Stopping": early_stopping,
                "FVAF_train": data.get("predict_training_eval", [None, None])[1], 
                "FVAF_val": data.get("predict_validation_eval", [None, None])[1],
                "FVAF_test": data.get("predict_testing_eval", [None, None])[1]
            })
    
    return pd.DataFrame(results).sort_values(["Ntraining", "Rotation"])

def find_best_hyperparameter(df, hyperparam_column):
    """
    Finds the best hyperparameter value based on max validation FVAF.
    
    :param df: DataFrame containing experimental results
    :param hyperparam_column: Column name (either 'dropout' or 'L2_regularization')
    :return: Best hyperparameter value
    """
    best_value = df.groupby(hyperparam_column)["FVAF_val"].mean().idxmax()
    return best_value

def compute_best_test_performance(df):
    """
    Computes the test performance for the best hyperparameter values.
    
    :param df: DataFrame containing experimental results
    :return: Dictionary of best test performances
    """
    best_dropout = find_best_hyperparameter(df[df["Dropout"].notna()], "Dropout")
    best_l2 = find_best_hyperparameter(df[df["L2_reg"].notna()], "L2_reg")

    results = {
        "No Regularization": df[df["Dropout"].isna() & df["L2_reg"].isna()].groupby("Ntraining")["FVAF_test"].mean(),
        "Early Stopping": df[df["Early_Stopping"] == True].groupby("Ntraining")["FVAF_test"].mean(),
        "Best Dropout": df[df["Dropout"] == best_dropout].groupby("Ntraining")["FVAF_test"].mean(),
        "Best Dropout + Early Stopping": df[(df["Dropout"] == best_dropout) & (df["Early_Stopping"] == True)].groupby("Ntraining")["FVAF_test"].mean(),
        "Best L2 Regularization": df[df["L2_reg"] == best_l2].groupby("Ntraining")["FVAF_test"].mean(),
    }
    
    return results, best_dropout, best_l2

def statistical_comparisons(df):
    """
    Computes paired t-tests comparing No Regularization with the other four models for Ntraining=1 and Ntraining=18.
    
    :param df: DataFrame containing test results
    """
    results, best_dropout, best_l2 = compute_best_test_performance(df)
    ntraining_cases = [1, 18]

    no_reg = load_results(["./part_1_pkl/"])
    early_stopping = load_results(["./results_part2/"])

    dropout_raw = load_results(["./results_part3/"])
    dropout = dropout_raw[dropout_raw["Dropout"] == best_dropout]

    dropout_early_stopping_raw = load_results(["./results_part4/"])
    dropout_early_stopping = dropout_early_stopping_raw[(dropout_early_stopping_raw["Dropout"] == best_dropout)]

    l2_reg_raw = load_results(["./results_part5/"])
    l2_reg = l2_reg_raw[l2_reg_raw["L2_reg"] == best_l2]

    comparisons = [early_stopping, dropout, dropout_early_stopping, l2_reg]

    for ntraining in ntraining_cases:
        no_reg_values = no_reg[no_reg["Ntraining"] == ntraining]["FVAF_test"]

        print(f"\nStatistical Comparisons for Ntraining = {ntraining}")

        for model in comparisons:
            model_values = model[model["Ntraining"] == ntraining]["FVAF_test"]

            if len(no_reg_values) > 0 and len(model_values) > 0:
                t_stat, p_val = ttest_rel(model_values, no_reg_values)
                mean_diff = model_values.mean() - no_reg_values.mean()
                
                print(f"t-statistic: {t_stat:.3f}, p-value = {p_val:.5f}, Mean Difference = {mean_diff:.5f}")

