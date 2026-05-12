import pandas as pd


# uniform weight across all live assets — 1/n each
def get_equal_weight_long_only(signal_row):
    signal_row = signal_row.dropna()
    n = len(signal_row)
    if n == 0:
        return pd.Series(dtype=float)
    weights = pd.Series(0.0, index=signal_row.index)
    weights.loc[signal_row.index] = 1.0 / n
    return weights


def compute_equal_weight_long_only_weights(signal_df):
    # apply per date along the asset axis
    return signal_df.apply(lambda row: get_equal_weight_long_only(row), axis=1).fillna(0.0)
