import pandas as pd


def get_weights(signal_row, long_quantile, short_quantile, target_net_exposure=0.0, target_gross_exposure=2.0):
    signal_row = signal_row.dropna()
    n = len(signal_row)
    if n == 0:
        return pd.Series(dtype=float)

    n_long, n_short = max(1, int(n * long_quantile)), max(1, int(n * short_quantile))
    ranked          = signal_row.sort_values()
    short_stocks    = ranked.index[:n_short]
    long_stocks     = ranked.index[-n_long:]

    weights = pd.Series(0.0, index=signal_row.index)

    long_target  = (target_gross_exposure + target_net_exposure) / 2
    short_target = (target_gross_exposure - target_net_exposure) / 2

    weights.loc[long_stocks] = long_target / n_long

    if short_target > 0:
        weights.loc[short_stocks] = -short_target / n_short

    return weights


# signal-prop L/S:  w_i = s_i / Σs_j  per side
# clip(0): top-q stock with s<0 → equal-wt fallback, never flip direction
def get_weights_signal_weighted(signal_row, long_quantile, short_quantile, target_net_exposure=0.0, target_gross_exposure=2.0):
    signal_row = signal_row.dropna()
    n = len(signal_row)
    if n == 0:
        return pd.Series(dtype=float)

    n_long, n_short = max(1, int(n * long_quantile)), max(1, int(n * short_quantile))
    ranked          = signal_row.sort_values()
    short_stocks    = ranked.index[:n_short]
    long_stocks     = ranked.index[-n_long:]

    weights = pd.Series(0.0, index=signal_row.index)

    long_signals  = signal_row.loc[long_stocks].clip(lower=0)
    short_signals = (-signal_row.loc[short_stocks]).clip(lower=0)

    long_target  = (target_gross_exposure + target_net_exposure) / 2
    short_target = (target_gross_exposure - target_net_exposure) / 2

    if long_signals.sum() <= 0:
        weights.loc[long_stocks] = long_target / n_long
    else:
        weights.loc[long_stocks] = long_target * (long_signals / long_signals.sum())

    if short_signals.sum() <= 0:
        weights.loc[short_stocks] = -short_target / n_short
    else:
        weights.loc[short_stocks] = -short_target * (short_signals / short_signals.sum())

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
