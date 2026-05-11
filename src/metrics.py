import numpy as np
from scipy import stats


# ann_ret = 12μ,  ann_vol = √12·σ,  Sharpe = ret/vol  (rf=0, mkt-neutral → justified)
def performance_metrics(returns):
    μ, σ    = returns.mean(), returns.std()
    ann_ret = μ * 12
    ann_vol = σ * np.sqrt(12)
    return ann_ret, ann_vol, ann_ret / ann_vol if ann_vol != 0 else np.nan


# DD(t) = NAV(t) / max_{s≤t} NAV(s) − 1
def drawdown(cumulative_returns):
    peak = cumulative_returns.cummax()
    dd   = cumulative_returns / peak - 1
    return dd, dd.min()


# Sortino = ann_ret / σ_down·√12   upside vol not penalised
def sortino_ratio(returns):
    ann_ret = returns.mean() * 12
    σ_down  = returns[returns < 0].std() * np.sqrt(12)
    return ann_ret / σ_down if σ_down != 0 else np.nan


# Calmar = ann_ret / |maxDD|  — how many times/yr does it earn back the worst hole
def calmar_ratio(returns, cumulative):
    _, max_dd = drawdown(cumulative)
    return returns.mean() * 12 / abs(max_dd) if max_dd != 0 else np.nan


# P(r > 0)
def hit_rate(returns):
    return (returns > 0).mean()


# H₀: μ=0   t = μ̂/(σ̂/√n)   |t|>2 → p<0.05 at n>30
def t_statistic(returns):
    t, p = stats.ttest_1samp(returns.dropna(), 0)
    return float(t), float(p)
