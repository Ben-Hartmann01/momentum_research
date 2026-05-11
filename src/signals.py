import numpy as np
import pandas as pd


# P(t-1)/P(t-L) − 1
def momentum_signal(prices, lookback):
    return prices.shift(1) / prices.shift(lookback) - 1


# μ_L / σ_ann — conviction per unit risk
# shift(1) on ret before rolling: σ(t) uses r[t-w…t-1] only, no bleed from r[t]
def risk_adjusted_momentum(prices, lookback, vol_window=12):
    raw_returns = prices.pct_change()
    rolling_vol = raw_returns.shift(1).rolling(vol_window).std() * np.sqrt(12)
    raw_mom     = momentum_signal(prices, lookback)
    return raw_mom / rolling_vol.where(rolling_vol != 0, np.nan)


# [z(μ_L) + z(μ_L/σ)] / 2
# z-score cross-sectionally first — scale kills the blend otherwise
def composite_signal(prices, lookback, vol_window=12):
    def _z(df):
        # z_i(t) = (x_i − μ_t) / σ_t across i
        m, s = df.mean(axis=1), df.std(axis=1)
        return df.sub(m, axis=0).div(s.where(s != 0, np.nan), axis=0)

<<<<<<< HEAD
    return (_z(momentum_signal(prices, lookback))
            + _z(risk_adjusted_momentum(prices, lookback, vol_window))) / 2
=======
def mean_reversion_signal(prices, lookback=1):
    return -(prices.shift(1) / prices.shift(1 + lookback) - 1) # -(P_i(t-1) / P_i(t-2) - 1); we assume some mean reversion after short-term price increasements

def composite_signal(prices, momentum_lookback=12, reversion_lookback=1, # combine signals
                     momentum_weight=0.7, reversion_weight=0.3):
    mom = momentum_signal(prices, momentum_lookback)
    rev = mean_reversion_signal(prices, reversion_lookback)

    # Z-scores: (Stock - mu(all stocks)) / sigma
    mom_z = mom.sub(mom.mean(axis=1), axis=0).div(mom.std(axis=1), axis=0)
    rev_z = rev.sub(rev.mean(axis=1), axis=0).div(rev.std(axis=1), axis=0) # for each date, take mean over all assets, subtract it from all assets; take sigma of all assets, divide

    return momentum_weight * mom_z + reversion_weight * rev_z
>>>>>>> b47b9cbf5522f25964ad6d8c81a101f302b84fbb
