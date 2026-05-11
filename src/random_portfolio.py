import pandas as pd
import numpy as np


# BM1: uniform draw from full universe, equal-wt each side  net=0, gross=2
def get_random_weights(signal_row, long_quantile, short_quantile, rng):
    signal_row = signal_row.dropna()
    n = len(signal_row)
    if n == 0:
        return pd.Series(dtype=float)

    n_long, n_short = max(1, int(n * long_quantile)), max(1, int(n * short_quantile))
    if n < n_long + n_short:
        return pd.Series(0.0, index=signal_row.index)

    pick         = rng.choice(signal_row.index.values, size=n_long + n_short, replace=False)
    weights      = pd.Series(0.0, index=signal_row.index)
    weights.loc[pick[:n_long]]  =  1.0 / n_long
    weights.loc[pick[n_long:]]  = -1.0 / n_short
    return weights


def compute_random_weights(signal_df, long_quantile, short_quantile):
    rng, rows = np.random.default_rng(), []
    for date in signal_df.index:
        w = get_random_weights(signal_df.loc[date], long_quantile, short_quantile, rng)
        w.name = date
        rows.append(w)
    return pd.DataFrame(rows).fillna(0.0)


# BM2: same draw, Dirichlet(1) weights ≡ uniform on simplex per side
# tests whether signal-prop weighting adds alpha over random concentration
def get_random_weights_signal_based(signal_row, long_quantile, short_quantile, rng):
    signal_row = signal_row.dropna()
    n = len(signal_row)
    if n == 0:
        return pd.Series(dtype=float)

    n_long, n_short = max(1, int(n * long_quantile)), max(1, int(n * short_quantile))
    if n < n_long + n_short:
        return pd.Series(0.0, index=signal_row.index)

    pick    = rng.choice(signal_row.index.values, size=n_long + n_short, replace=False)
    lw      = rng.random(n_long);  lw  /= lw.sum()
    sw      = rng.random(n_short); sw  /= sw.sum()

    weights = pd.Series(0.0, index=signal_row.index)
    weights.loc[pick[:n_long]] =  lw
    weights.loc[pick[n_long:]] = -sw
    return weights


def compute_random_weights_signal_based(signal_df, long_quantile, short_quantile):
    rng, rows = np.random.default_rng(), []
    for date in signal_df.index:
        w = get_random_weights_signal_based(signal_df.loc[date], long_quantile, short_quantile, rng)
        w.name = date
        rows.append(w)
    return pd.DataFrame(rows).fillna(0.0)
