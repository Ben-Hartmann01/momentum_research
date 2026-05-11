# Quantitative Momentum Strategy — Walk-Forward Backtesting

**Status:** active development — results not production-ready ()

---

## Objective

Market-neutral long/short equity strategy on cross-sectional momentum.
Walk-forward backtesting framework with signal evaluation, parameter selection, and statistical significance.

---

<<<<<<< HEAD
## Signals
=======
## Strategies Description

The current 2 compared strategies are cross-sectional momentum/mean_reversion strategies:
1. Equal-weight Strategy
   1. A signal is computed based on past price performance over a given lookback window 

   2. Assets are ranked according to the signal
      * we use a momentum signal and a momentum x mean reversion signal and compare both
>>>>>>> b47b9cbf5522f25964ad6d8c81a101f302b84fbb

Three signals, all skip-1-month to avoid microstructure reversal:

<<<<<<< HEAD
| Signal | Formula at time t |
|---|---|
| Raw Momentum | `P(t-1)/P(t-L) - 1` |
| Risk-Adj Momentum | `μ_L(t) / σ_ann(t-1)` — vol computed through t-1 only, no look-ahead |
| Composite | `[z(μ_L) + z(μ_L/σ)] / 2` — cross-sectional z-score before blending |

---
=======
      * top quantile is taken long
      * bottom quantile is taken short
      * everything in between stays neutral
      * positions are equal-weighted within each side
      * Net exposures 0, 0.5, 1 get tested (gross expsoure = 2) -> weights within each direction are equal, but cross-directional only in the case of net exposure being 0

   4. The portfolio is rebalanced on a monthly basis (ME)

2. Signal-based-weight Strategy
   1. A signal is computed based on past price performance over a given lookback window 

   2. Assets are ranked according to the signal
      * we use a momentum signal and a momentum x mean reversion signal and compare both
>>>>>>> b47b9cbf5522f25964ad6d8c81a101f302b84fbb

## Portfolio Construction

<<<<<<< HEAD
Signal-weighted, market-neutral long/short:
=======
      * top quantile assets are taken as long candidates
      * bottom quantile assets are taken as short candidates
      * everything in between stays neutral
      * positions are weighted based on their normalized signal (signal / sum(signals) in this quantile)
      * Net exposures 0, 0.5, 1 get tested (gross expsoure = 2) 
>>>>>>> b47b9cbf5522f25964ad6d8c81a101f302b84fbb

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

<<<<<<< HEAD
---

## Benchmark
=======
Buy and hold strategy as Benchmark

* every asset gets weight 1 / n
* hold each position until the end of the maturity
* we therefore have no transaction costs
>>>>>>> b47b9cbf5522f25964ad6d8c81a101f302b84fbb

Random long/short with identical gross/net exposure:

<<<<<<< HEAD
- Select `n_long + n_short` assets uniformly from full universe (no signal conditioning)
- Weights drawn from Dirichlet(1) ≡ uniform on simplex per side
- Reuses same `(L*, q*)` as strategy → isolates weighting skill from param selection

=======
>>>>>>> b47b9cbf5522f25964ad6d8c81a101f302b84fbb
---

## Project Structure

```
src/
<<<<<<< HEAD
├── data_loader.py       — yfinance download, monthly resample
├── signals.py           — momentum_signal, risk_adjusted_momentum, composite_signal
├── ic_analysis.py       — compute_ic, compute_icir, ic_summary, compute_ic_decay
├── portfolio.py         — equal-weight and signal-weighted construction
├── backtest.py          — lagged-weight return computation, transaction costs
├── metrics.py           — Sharpe, Sortino, Calmar, hit_rate, t_statistic, drawdown
└── random_portfolio.py  — random benchmarks

main.py                  — data load → IC analysis → walk-forward → plots
=======
├── data_loader.py
├── signals.py
├── portfolio.py
├── backtest.py
├── benchmark.py
├── metrics.py
├── random_portfolio.py

main.py
>>>>>>> b47b9cbf5522f25964ad6d8c81a101f302b84fbb
```

---

## Known Limitations

<<<<<<< HEAD
- Signal is extremely weak. In returns and also IC, mainly due to vola weighting
- Transaction cost model: flat 10bps per unit turnover (no market impact, no bid-ask)
- Benchmark picks from full universe — not conditioned on signal top/bottom quantile
- Universe: ~90 US large-cap stocks — survivorship bias (all currently in S&P 500)
- Signal-weighted momentum is sensitive to momentum crashes (2020, 2022): concentrates in the exact stocks that reverse hardest in reversal episodes
=======
* simplified transaction cost model 
* weak benchmark (Its tough to not outperform the first BM)
* no statistical significance testing
* there is an obvious improvement of the strategies due to more net exposure
>>>>>>> b47b9cbf5522f25964ad6d8c81a101f302b84fbb

---

## Planned

<<<<<<< HEAD
- Improve the signal
- Volatility-scaled position sizing — reduces crash exposure
- Walk-forward IC analysis — detect signal decay across time
- Stronger benchmark: random selection restricted to top/bottom quantile
=======
* clean plot
* improved transaction cost modeling (check realistic values)
* different benchmarks 
* additional signals
* improved reporting and visualization
* get rid of the market exposure bias
>>>>>>> b47b9cbf5522f25964ad6d8c81a101f302b84fbb

---

## Setup

```bash
pip install pandas numpy scipy yfinance matplotlib
python main.py
```
