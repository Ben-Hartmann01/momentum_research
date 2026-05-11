import pandas as pd

def get_equal_weight_long_only(signal_row):
    signal_row = signal_row.dropna()

    n = len(signal_row)
    if n == 0:
        return pd.Series(dtype = float)

    weights = pd.Series(0.0, index=signal_row.index)
    weights.loc[signal_row.index] = 1.0 / n # every asset gets he same weight s.t. they sum to 1

    return weights

def compute_equal_weight_long_only_weights(signal_df):
    return signal_df.apply(lambda row: get_equal_weight_long_only(row), axis=1).fillna(0.0) # along the assets (axis 1) apply equal weights




