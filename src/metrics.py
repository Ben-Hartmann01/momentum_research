import numpy as np
from scipy import stats


# monthly → annualized: 12x return, √12 vol; rf = 0 (ok for L/S)
def performance_metrics(returns):
    μ, σ    = returns.mean(), returns.std()
    ann_ret = μ * 12
    ann_vol = σ * np.sqrt(12)
    return ann_ret, ann_vol, ann_ret / ann_vol if ann_vol != 0 else np.nan


# NAV path relative to its running peak
def drawdown(cumulative_returns):
    peak = cumulative_returns.cummax()
    dd   = cumulative_returns / peak - 1
    return dd, dd.min()


# like Sharpe but only penalizes downside vol
def sortino_ratio(returns):
    ann_ret = returns.mean() * 12
    σ_down  = returns[returns < 0].std() * np.sqrt(12)
    return ann_ret / σ_down if σ_down != 0 else np.nan


# how many times per year does it earn back its worst hole
def calmar_ratio(returns, cumulative):
    _, max_dd = drawdown(cumulative)
    return returns.mean() * 12 / abs(max_dd) if max_dd != 0 else np.nan


def hit_rate(returns):
    return (returns > 0).mean()


# H₀: E[r] = 0; |t| > 2 → roughly significant at n > 30
def t_statistic(returns):
    t, p = stats.ttest_1samp(returns.dropna(), 0)
    return float(t), float(p)
