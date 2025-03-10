import pandas as pd
import numpy as np
from scipy.stats import rankdata


def transform(files):
    true = pd.DataFrame()
    predict_mean = pd.DataFrame()
    i=0
    for file in files:
        df = pd.read_pickle(file)
        df["true"] = (df["true"] - df["true"].mean()) / df["true"].std()
        df['mean_prediction'] = (df['mean_prediction'] - df['mean_prediction'].mean()) / df['mean_prediction'].std()
        true[f'{i}'] = df['true']
        predict_mean[f'{i}'] = df['mean_prediction']
        i+=1
    return true, predict_mean

def quantile_normalize(df):
    """Applies quantile normalization to a DataFrame."""
    sorted_df = np.sort(df, axis=0)  # Sort each column
    rank_mean = np.mean(sorted_df, axis=1)  # Compute mean across columns
    ranks = np.apply_along_axis(rankdata, 0, df) - 1  # Get ranks (0-based)
    norm_df = np.zeros_like(df)
    
    for i in range(df.shape[1]):
        norm_df[:, i] = rank_mean[ranks[:, i].astype(int)]  # Assign mean rank values
    
    return pd.DataFrame(norm_df, index=df.index, columns=df.columns)




