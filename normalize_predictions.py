import pandas as pd
import numpy as np


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




