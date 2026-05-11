import pandas as pd


# equal-wt L/S:  w_i = ±1/n_{L/S}   net=0, gross=2
def get_weights(signal_row, long_quantile, short_quantile):
    signal_row = signal_row.dropna()
    n = len(signal_row)
    if n == 0:
        return pd.Series(dtype=float)

    n_long, n_short = max(1, int(n * long_quantile)), max(1, int(n * short_quantile))
    ranked          = signal_row.sort_values()
    short_stocks    = ranked.index[:n_short]
    long_stocks     = ranked.index[-n_long:]

    weights = pd.Series(0.0, index=signal_row.index)
    weights.loc[long_stocks]  =  1.0 / n_long
    weights.loc[short_stocks] = -1.0 / n_short
    return weights


def compute_weights(signal_df, long_quantile, short_quantile):
    rows = []
    for date in signal_df.index:
        w = get_weights(signal_df.loc[date], long_quantile, short_quantile)
        w.name = date
        rows.append(w)
    return pd.DataFrame(rows).fillna(0.0)


# signal-prop L/S:  w_i = s_i / Σs_j  per side
# clip(0): top-q stock with s<0 → equal-wt fallback, never flip direction
def get_weights_signal_weighted(signal_row, long_quantile, short_quantile):
    signal_row = signal_row.dropna()
    n = len(signal_row)
    if n == 0:
        return pd.Series(dtype=float)

    n_long, n_short = max(1, int(n * long_quantile)), max(1, int(n * short_quantile))
    ranked          = signal_row.sort_values()
    short_stocks    = ranked.index[:n_short]
    long_stocks     = ranked.index[-n_long:]

    weights = pd.Series(0.0, index=signal_row.index)
    ls = signal_row.loc[long_stocks].clip(lower=0)
    ss = (-signal_row.loc[short_stocks]).clip(lower=0)

    weights.loc[long_stocks]  = ls / ls.sum() if ls.sum() > 0 else  1.0 / n_long
    weights.loc[short_stocks] = -ss / ss.sum() if ss.sum() > 0 else -1.0 / n_short
    return weights


def compute_weights_signal_weighted(signal_df, long_quantile, short_quantile):
    rows = []
    for date in signal_df.index:
        w = get_weights_signal_weighted(signal_df.loc[date], long_quantile, short_quantile)
        w.name = date
        rows.append(w)
    return pd.DataFrame(rows).fillna(0.0)


def check_portfolio(weights, name):
    print(name)
    print("avg long  :", weights.clip(lower=0).sum(axis=1).mean())
    print("avg short :", (-weights.clip(upper=0)).sum(axis=1).mean())
    print("avg net   :", weights.sum(axis=1).mean())
