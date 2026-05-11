import pandas as pd

<<<<<<< HEAD

# equal-wt L/S:  w_i = ±1/n_{L/S}   net=0, gross=2
def get_weights(signal_row, long_quantile, short_quantile):
=======
# basic fixed number, same weight approach
def get_weights(signal_row, long_quantile, short_quantile, target_net_exposure=0.0, target_gross_exposure=2.0):
>>>>>>> b47b9cbf5522f25964ad6d8c81a101f302b84fbb
    signal_row = signal_row.dropna()
    n = len(signal_row)
    if n == 0:
        return pd.Series(dtype=float)
<<<<<<< HEAD

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
=======

    n_long = max(1, int(n * long_quantile))
    n_short = max(1, int(n * short_quantile))

    ranked = signal_row.sort_values()

    short_stocks = ranked.index[:n_short]
    long_stocks = ranked.index[-n_long:]

    weights = pd.Series(0.0, index=signal_row.index)

    long_target = (target_gross_exposure + target_net_exposure) / 2
    short_target = (target_gross_exposure - target_net_exposure) / 2

    weights.loc[long_stocks] = long_target / n_long

    if short_target > 0:
        weights.loc[short_stocks] = -short_target / n_short

    return weights

# signal-weighted approach
def get_weights_signal_weighted(signal_row, long_quantile, short_quantile, target_net_exposure=0.0, target_gross_exposure=2.0):
>>>>>>> b47b9cbf5522f25964ad6d8c81a101f302b84fbb
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

<<<<<<< HEAD
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

=======
    # make sure signs are positive / negative --> no "wrong" moves
    long_signals = signal_row.loc[long_stocks].clip(lower=0)
    short_signals = (-signal_row.loc[short_stocks]).clip(lower=0)

    long_target = (target_gross_exposure + target_net_exposure) / 2
    short_target = (target_gross_exposure - target_net_exposure) / 2 # allow different net exposures

    # fallback in case sums are zero or negative for some reason
    if long_signals.sum() <= 0:
        weights.loc[long_stocks] = long_target / n_long
    else:
        weights.loc[long_stocks] = long_target * (long_signals / long_signals.sum())

    if short_signals.sum() <= 0:
        weights.loc[short_stocks] = - short_target / n_short
    else:
        weights.loc[short_stocks] = - short_target * (short_signals / short_signals.sum())

    return weights

def _compute_weights_generic(weight_fn, signal_df, long_quantile, short_quantile,
                              target_net_exposure=0.0, target_gross_exposure=2.0):
    weights_list = []
    for date in signal_df.index:
        w = weight_fn(signal_df.loc[date], long_quantile, short_quantile,
                      target_net_exposure, target_gross_exposure)
        w.name = date
        weights_list.append(w)
    return pd.DataFrame(weights_list).fillna(0.0)
>>>>>>> b47b9cbf5522f25964ad6d8c81a101f302b84fbb

def compute_weights(signal_df, long_quantile, short_quantile,
                    target_net_exposure=0.0, target_gross_exposure=2.0):
    return _compute_weights_generic(get_weights, signal_df, long_quantile, short_quantile,
                                    target_net_exposure, target_gross_exposure)

def compute_weights_signal_weighted(signal_df, long_quantile, short_quantile,
                                    target_net_exposure=0.0, target_gross_exposure=2.0):
    return _compute_weights_generic(get_weights_signal_weighted, signal_df, long_quantile,
                                    short_quantile, target_net_exposure, target_gross_exposure)

def check_portfolio(weights, name):
    print(name)
    print("avg long  :", weights.clip(lower=0).sum(axis=1).mean())
    print("avg short :", (-weights.clip(upper=0)).sum(axis=1).mean())
    print("avg net   :", weights.sum(axis=1).mean())
