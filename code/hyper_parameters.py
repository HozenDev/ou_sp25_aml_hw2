from scipy.stats import ttest_rel

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
    best_dropout = find_best_hyperparameter(df[df["dropout"].notna()], "dropout")
    best_l2 = find_best_hyperparameter(df[df["L2_regularization"].notna()], "L2_regularization")

    results = {
        "No Regularization": df[df["dropout"].isna() & df["L2_regularization"].isna()].groupby("Ntraining")["FVAF_test"].mean(),
        "Early Stopping": df[df["early_stopping"] == True].groupby("Ntraining")["FVAF_test"].mean(),
        "Best Dropout": df[df["dropout"] == best_dropout].groupby("Ntraining")["FVAF_test"].mean(),
        "Best Dropout + Early Stopping": df[(df["dropout"] == best_dropout) & (df["early_stopping"] == True)].groupby("Ntraining")["FVAF_test"].mean(),
        "Best L2 Regularization": df[df["L2_regularization"] == best_l2].groupby("Ntraining")["FVAF_test"].mean(),
    }
    
    return results, best_dropout, best_l2

def statistical_comparisons(df):
    """
    Computes paired t-tests comparing No Regularization with the other four models for Ntraining=1 and Ntraining=18.
    
    :param df: DataFrame containing test results
    """
    results, _, _ = compute_best_test_performance(df)

    ntraining_cases = [1, 18]
    comparisons = ["Early Stopping", "Best Dropout", "Best Dropout + Early Stopping", "Best L2 Regularization"]

    for ntraining in ntraining_cases:
        no_reg_values = df[(df["Ntraining"] == ntraining) & df["dropout"].isna() & df["L2_regularization"].isna()]["FVAF_test"]

        print(f"\nStatistical Comparisons for Ntraining = {ntraining}")

        for model in comparisons:
            model_values = df[(df["Ntraining"] == ntraining) & df[model.replace(" ", "_").lower()].notna()]["FVAF_test"]
            
            if len(no_reg_values) > 0 and len(model_values) > 0:
                t_stat, p_val = ttest_rel(no_reg_values, model_values)
                mean_diff = model_values.mean() - no_reg_values.mean()
                
                print(f"{model}: p-value = {p_val:.5f}, Mean Difference = {mean_diff:.5f}")

