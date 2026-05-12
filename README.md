# momentum_research

cross-sectional equity momentum / mean-reversion backtest.
monthly frequency, walk-forward, IC-weighted signals.

**status:** active research — signals still weak and very noisy, not production-ready.

---

## what it does

- computes 4 signals on ~85 US large-caps
- evaluates signal quality via IC / ICIR / IC decay (full sample, no fitted params)
- runs 12 portfolio strategies through a walk-forward (36m train → 6m test, rolling)
- outputs a single metrics table + cumulative return plots for the exact same strategy set

---

## signals

all signals use skip-1 (`prices.shift(1)`) so signal at `t` never touches `r_t`.

| signal | formula |
|---|---|
| momentum | `P(t-1) / P(t-L) - 1` |
| risk-adj momentum | `mom_L / σ_ann(t-1)` — σ rolled on lagged returns |
| mean reversion | `-(P(t-1) / P(t-2) - 1)` — always 1-month reversal |
| composite | IC-weighted z-score blend of all three above |

### composite signal

three components, each cross-sectionally z-scored.
weights at `t` = rolling 24m mean absolute IC per component, shifted by 1 period (so no future IC leaks in).
normalized to sum to 1; equal weights (1/3 each) as fallback until IC history accumulates.

```
composite(t) = w_mom(t) * z_mom(t) + w_ram(t) * z_ram(t) + w_rev(t) * z_rev(t)
w_k(t) = |rolling_IC_k(t-1)| / Σ|rolling_IC_j(t-1)|
```

cross-sectional z-score everywhere — scale differences would otherwise dominate the blend.

---

## registries (central source of truth)

two dicts drive everything. no other place controls what runs.

**`SIGNAL_REGISTRY`** — 4 entries, drives IC analysis only:

```python
{
    "Momentum":       momentum_signal,
    "Risk-Adj Mom":   risk_adjusted_momentum,
    "Mean Reversion": lambda p, lb: mean_reversion_signal(p, lookback=1),
    "Composite":      composite_signal,
}
```

**`PORTFOLIO_CONFIGS`** — 12 entries, drives walk-forward + metrics table + plots:

```
Momentum   × {equal, signal-wt} × {L/S, half-L/S, long-only}   →  6 strategies
Composite  × {equal, signal-wt} × {L/S, half-L/S, long-only}   →  6 strategies
```

adding a strategy = one row in `PORTFOLIO_CONFIGS`.
adding a signal to IC analysis = one key in `SIGNAL_REGISTRY`.

---

## portfolio construction

both methods rank assets cross-sectionally on the signal and select top/bottom `q` quantile.

**equal-weight:**
`w_i = long_target / n_long` (long side), `-short_target / n_short` (short side)

**signal-weighted:**
`w_i ∝ s_i` within the selected quantile — proportional to signal strength.
fallback to equal weights if all candidates have wrong sign.

exposure arithmetic:
```
long_target  = (gross + net) / 2
short_target = (gross - net) / 2
gross = 2 always
```

net exposures tested: `{0.0, 0.5, 1.0}` — market-neutral, half-directional, long-only.

---

## walk-forward

```
train 36m → grid search → test 6m → roll +6m → repeat
```

grid: `lookback ∈ {3, 6, 9, 12}`, `q ∈ {0.1, 0.2, 0.3, 0.4, 0.5}` → 20 combos per fold.
param selection: max in-sample Sharpe.
signal computed on `[0, test_end]` so rolling windows have warmup — only eval window is OOS.

---

## signal quality (IC / ICIR)

run on full sample before any fitting — no look-ahead.

| metric | def |
|---|---|
| IC(t) | `spearman_rank_corr(signal_t, r_{t→t+1})` |
| ICIR | `mean(IC) / std(IC)` — signal Sharpe |
| IC decay | mean IC at horizons h=1..6m using cumulative return `P(t+h)/P(t) - 1` |

rough thresholds: `|mean IC| > 0.05` → live signal, `ICIR > 0.5` → production-grade.

all 4 signals evaluated.

---

## benchmark

equal-weight long-only across the full universe.
`w_i = 1/n` per asset at each rebalance.
transaction costs applied (same 10bps model as strategies).
computed once via the same fold structure as the walk-forward — not a buy-and-hold series.

---

## performance metrics

output as a single pandas DataFrame table — all 12 strategies + benchmark in one place.

| metric   | def                                  |
|----------|--------------------------------------|
| Sharpe   | `ann_ret / ann_vol`, rf = 0          |
| Sortino  | `ann_ret / σ_downside`               |
| Calmar   | `ann_ret / abs(max_DD)`              |
| Hit Rate | `P(r_t > 0)`                         |
| t-stat   | `H₀: E[r] = 0`, `                    |t| > 2 → p < 0.05` |
| Max DD   | `min_t(NAV_t / max_{s≤t} NAV_s - 1)` |

---

## transaction costs

flat 10bps per unit of portfolio turnover:
```
cost(t) = 10bps × Σ_i |w_i(t) - w_i(t-1)|
```

no market impact, no spread model.

---

## output

1. IC/ICIR/decay table printed to stdout (all 4 signals)
2. IC time series plot (rolling 12m mean IC) + IC decay bar chart
3. metrics table (all 12 strategies + benchmark, one DataFrame)
4. cumulative return plot — 2 panels: equal-weighted strategies | signal-weighted strategies, benchmark in both

---

## known issues

- survivorship bias — universe is ~85 current large-caps, no delisted names
- signals weak in practice, IC often below threshold
- vol scaling in risk-adj momentum can suppress signal in high-vol regimes
- TC model simplified — 10bps flat understates real costs at this turnover
- random L/S benchmark exists (`src/random_portfolio.py`) but not wired into default run
- higher net exposure mechanically picks up market beta — not pure signal comparison

---

## structure

```
src/
  data_loader.py      yfinance download, monthly resampling
  signals.py          momentum, risk-adj momentum, mean reversion, composite
  ic_analysis.py      IC, ICIR, IC decay
  portfolio.py        equal-weight and signal-weighted construction
  backtest.py         lagged-weight returns, transaction costs
  benchmark.py        equal-weight long-only benchmark weights
  metrics.py          Sharpe, Sortino, Calmar, hit rate, t-stat, drawdown
  random_portfolio.py random L/S benchmark (available, not in default run)

main.py               data load → IC analysis → walk-forward → table + plots
```

---

## setup

```
pip install pandas numpy scipy matplotlib yfinance
python main.py
```
