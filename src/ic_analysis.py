import pandas as pd
import numpy as np


# IC(t) := rank_corr(s_t, r_{t+1})  Spearman, cross-sectional
# |mean IC| > 0.05 → live signal
def compute_ic(signal_df, forward_returns):
    common_dates = signal_df.index.intersection(forward_returns.index)
    ic_values    = {}

    for date in common_dates:
        sig    = signal_df.loc[date].dropna()
        ret    = forward_returns.loc[date].dropna()
        common = sig.index.intersection(ret.index)

        if len(common) < 5:   # rank corr breaks down at small n
            ic_values[date] = np.nan
            continue

        ic_values[date] = sig.loc[common].rank().corr(ret.loc[common].rank())

    return pd.Series(ic_values)


# ICIR := mean(IC) / std(IC)  — Sharpe of the signal
# > 0.5 → production-grade consistency
def compute_icir(ic_series):
    ic = ic_series.dropna()
    return ic.mean() / ic.std() if ic.std() != 0 else np.nan


def ic_summary(ic_series, signal_name="Signal"):
    ic, icir = ic_series.dropna(), compute_icir(ic_series)
    print(f"\n--- IC: {signal_name} ---")
    print(f"  mean IC  : {ic.mean():.4f}   (>0.05 live)")
    print(f"  IC std   : {ic.std():.4f}")
    print(f"  ICIR     : {icir:.4f}   (>0.5 prod)")
    print(f"  IC>0 (%) : {(ic > 0).mean()*100:.1f}%")
    print(f"  |IC|>0.05: {(ic.abs() > 0.05).mean()*100:.1f}%")
    return {"mean_ic": ic.mean(), "ic_std": ic.std(), "icir": icir}


# decay at h: IC(s_t, P(t+h)/P(t)−1)  — cumulative, not period
# period IC at h is noise: no reason s_t predicts r_{t+h} in isolation
def compute_ic_decay(prices, signal_func, max_horizon=6, lookback=12):
    signals = signal_func(prices, lookback)
    decay   = {}

    for h in range(1, max_horizon + 1):
        fwd      = prices.shift(-h) / prices - 1   # P(t+h)/P(t) − 1
        decay[h] = compute_ic(signals, fwd).dropna().mean()

    return pd.Series(decay)
