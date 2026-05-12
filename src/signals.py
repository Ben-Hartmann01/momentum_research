import numpy as np
import pandas as pd


# skip-1: P(t-1)/P(t-L) so signal at t never touches r_t
def momentum_signal(prices, lookback):
    return prices.shift(1) / prices.shift(lookback) - 1


# basically Sharpe of the lookback window — scales out vol differences across assets
# shift(1) on raw returns: σ(t) only uses r[t-w…t-1], no bleed from r[t]
def risk_adjusted_momentum(prices, lookback, vol_window=12):
    raw_returns = prices.pct_change()
    rolling_vol = raw_returns.shift(1).rolling(vol_window).std() * np.sqrt(12)
    return momentum_signal(prices, lookback) / rolling_vol.where(rolling_vol > 0, np.nan)


# 1-month reversal — negate the most recent return
def mean_reversion_signal(prices, lookback=1):
    return -(prices.shift(1) / prices.shift(1 + lookback) - 1)


# cross-sec normalize: subtract cross-sec mean, divide by cross-sec std
def _cs_zscore(df):
    m = df.mean(axis=1)
    s = df.std(axis=1)
    return df.sub(m, axis=0).div(s.where(s > 0, np.nan), axis=0)


# one IC per date, then rolling mean over ic_window months
def _rolling_ic(signal_df, fwd_returns, ic_window):
    common_dates = signal_df.index.intersection(fwd_returns.index)
    ic_vals = {}
    for date in common_dates:
        sig    = signal_df.loc[date].dropna()
        ret    = fwd_returns.loc[date].dropna()
        common = sig.index.intersection(ret.index)
        if len(common) >= 5:
            ic_vals[date] = sig.loc[common].rank().corr(ret.loc[common].rank())
    ic_series = pd.Series(ic_vals)
    return ic_series.rolling(ic_window, min_periods=max(3, ic_window // 2)).mean()


# IC-weighted blend of mom, risk-adj mom, 1m reversal
# all three cross-sec z-scored first — otherwise scale kills the blend
def composite_signal(prices, lookback=12, vol_window=12, rev_lookback=1, ic_window=24):
    fwd_returns = prices.pct_change().shift(-1)  # r_{t+1} — used to estimate IC

    mom = _cs_zscore(momentum_signal(prices, lookback))
    ram = _cs_zscore(risk_adjusted_momentum(prices, lookback, vol_window))
    rev = _cs_zscore(mean_reversion_signal(prices, rev_lookback))

    # shift(1) so weight at t uses IC only up to t-1 — no look-ahead
    ic_mom = _rolling_ic(mom, fwd_returns, ic_window).shift(1)
    ic_ram = _rolling_ic(ram, fwd_returns, ic_window).shift(1)
    ic_rev = _rolling_ic(rev, fwd_returns, ic_window).shift(1)

    # absolute IC as weight — signals already point in the right direction
    ic_w = pd.DataFrame(
        {'mom': ic_mom.abs(), 'ram': ic_ram.abs(), 'rev': ic_rev.abs()}
    ).reindex(prices.index)

    row_sum = ic_w.sum(axis=1)
    # normalize; if no IC history yet, fall back to 1/3 each
    ic_w = ic_w.div(row_sum.where(row_sum > 0, np.nan), axis=0).fillna(1.0 / 3)

    return (
        mom.mul(ic_w['mom'], axis=0)
        + ram.mul(ic_w['ram'], axis=0)
        + rev.mul(ic_w['rev'], axis=0)
    )
