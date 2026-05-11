# Quantitative Momentum Strategy — Walk-Forward Backtesting

**Status:** active development — results not production-ready ()

---

## Objective

Market-neutral long/short equity strategy on cross-sectional momentum.
Walk-forward backtesting framework with signal evaluation, parameter selection, and statistical significance.

---

## Signals

Three signals, all skip-1-month to avoid microstructure reversal:

| Signal | Formula at time t |
|---|---|
| Raw Momentum | `P(t-1)/P(t-L) - 1` |
| Risk-Adj Momentum | `μ_L(t) / σ_ann(t-1)` — vol computed through t-1 only, no look-ahead |
| Composite | `[z(μ_L) + z(μ_L/σ)] / 2` — cross-sectional z-score before blending |

---

## Portfolio Construction

Signal-weighted, market-neutral long/short:

- Top-q quantile → long candidates; bottom-q → short candidates
- `w_i = s_i / Σs_j` within each side (signal-proportional weights)
- Fallback to equal-weight if all candidate signals have wrong sign
- `net = 0`, `gross = 2`; rebalanced monthly

---

## Walk-Forward Backtesting

No look-ahead bias, no reuse of test data:

- Train on fixed 36-month window → grid search `(lookback ∈ {3,6,9,12}, q ∈ {0.1,...,0.5})`
- Select `(L*, q*)` maximising in-sample Sharpe
- Test on next 6 months OOS using `(L*, q*)`
- Roll forward by 6 months, repeat

---

## Signal Quality: IC / ICIR

Evaluated before backtesting on full sample (no params fitted → no look-ahead):

- `IC(t) = rank_corr(signal_t, r_{t→t+1})` — Spearman, cross-sectional
- `ICIR = mean(IC) / std(IC)` — Sharpe of the signal
- IC decay at `h = 1..6m` uses `P(t+h)/P(t)-1` (cumulative, not individual period returns)

Thresholds: `|mean IC| > 0.05` useful; `ICIR > 0.5` production-grade

---

## Performance Metrics

| Metric | Definition |
|---|---|
| Sharpe | `ann_ret / ann_vol` (rf = 0, justified for market-neutral) |
| Sortino | `ann_ret / downside_std` — upside vol not penalised |
| Calmar | `ann_ret / |max_DD|` |
| Hit rate | `P(r_t > 0)` |
| t-statistic | `H₀: E[r]=0`; `|t|>2 ≈ p<0.05` for n>30 |
| Max drawdown | `min_t { NAV(t) / max_{s≤t} NAV(s) - 1 }` |

---

## Benchmark

Random long/short with identical gross/net exposure:

- Select `n_long + n_short` assets uniformly from full universe (no signal conditioning)
- Weights drawn from Dirichlet(1) ≡ uniform on simplex per side
- Reuses same `(L*, q*)` as strategy → isolates weighting skill from param selection

---

## Project Structure

```
src/
├── data_loader.py       — yfinance download, monthly resample
├── signals.py           — momentum_signal, risk_adjusted_momentum, composite_signal
├── ic_analysis.py       — compute_ic, compute_icir, ic_summary, compute_ic_decay
├── portfolio.py         — equal-weight and signal-weighted construction
├── backtest.py          — lagged-weight return computation, transaction costs
├── metrics.py           — Sharpe, Sortino, Calmar, hit_rate, t_statistic, drawdown
└── random_portfolio.py  — random benchmarks

main.py                  — data load → IC analysis → walk-forward → plots
```

---

## Known Limitations

- Signal is extremely weak. In returns and also IC, mainly due to vola weighting
- Transaction cost model: flat 10bps per unit turnover (no market impact, no bid-ask)
- Benchmark picks from full universe — not conditioned on signal top/bottom quantile
- Universe: ~90 US large-cap stocks — survivorship bias (all currently in S&P 500)
- Signal-weighted momentum is sensitive to momentum crashes (2020, 2022): concentrates in the exact stocks that reverse hardest in reversal episodes

---

## Planned

- Improve the signal
- Volatility-scaled position sizing — reduces crash exposure
- Walk-forward IC analysis — detect signal decay across time
- Stronger benchmark: random selection restricted to top/bottom quantile

---

## Setup

```bash
pip install pandas numpy scipy yfinance matplotlib
python main.py
```
