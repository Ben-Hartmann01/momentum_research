# Quantitative Momentum Strategy — Walk-Forward Backtesting

**Status:** active development — results not production-ready (returns and IC still very weak)

---

## Objective

Market-neutral long/short equity strategy based on cross-sectional momentum.

The project implements a walk-forward backtesting framework with signal evaluation, parameter selection, benchmark comparison, transaction costs, and statistical performance metrics.

---

## Signals

The current strategies use cross-sectional momentum and mean-reversion-related momentum signals.

All signals use a skip-1-month structure to reduce microstructure reversal effects.

| Signal | Formula at time t |
|---|---|
| Raw Momentum | `P(t-1) / P(t-L) - 1` |
| Risk-Adjusted Momentum | `μ_L(t) / σ_ann(t-1)` — volatility computed through `t-1` only |
| Composite | `[z(μ_L) + z(μ_L / σ)] / 2` — cross-sectional z-score before blending |

---

## Portfolio Construction

Two portfolio construction approaches are compared.

### 1. Equal-Weight Long/Short Strategy

1. A signal is computed based on past price performance over a given lookback window.
2. Assets are ranked according to the signal.
3. The top quantile is taken long.
4. The bottom quantile is taken short.
5. Assets between both quantiles remain neutral.
6. Positions are equal-weighted within each side.
7. The portfolio is rebalanced monthly.

Tested exposure settings:

- `net = 0`
- `net = 0.5`
- `net = 1`
- `gross = 2`

### 2. Signal-Weighted Long/Short Strategy

1. A signal is computed based on past price performance over a given lookback window.
2. Assets are ranked according to the signal.
3. The top quantile is taken long.
4. The bottom quantile is taken short.
5. Assets between both quantiles remain neutral.
6. Positions are weighted based on normalized signal strength within each side.
7. The portfolio is rebalanced monthly.

Weighting rule:

`w_i = s_i / Σs_j`

Fallback:

- If all candidate signals have the wrong sign, the portfolio falls back to equal weights.

Default exposure setting:

- `net = 0`
- `gross = 2`

---

## Walk-Forward Backtesting

The backtest avoids look-ahead bias and does not reuse test data for parameter selection.

Procedure:

1. Train on a fixed 36-month window.
2. Run grid search over:
   - `lookback ∈ {3, 6, 9, 12}`
   - `q ∈ {0.1, 0.2, 0.3, 0.4, 0.5}`
3. Select `(L*, q*)` that maximizes in-sample Sharpe.
4. Test the selected parameters on the next 6 months out-of-sample.
5. Roll forward by 6 months and repeat.

---

## Signal Quality: IC / ICIR

Signal quality is evaluated before backtesting on the full sample.

Since no parameters are fitted during this step, this does not introduce look-ahead bias.

| Metric | Definition |
|---|---|
| IC | `rank_corr(signal_t, r_{t→t+1})` — Spearman cross-sectional rank correlation |
| ICIR | `mean(IC) / std(IC)` — Sharpe-like measure of signal quality |
| IC decay | Evaluated for horizons `h = 1..6m` using `P(t+h) / P(t) - 1` |

Interpretation:

- `|mean IC| > 0.05` may indicate a useful signal
- `ICIR > 0.5` is considered strong

---

## Performance Metrics

| Metric | Definition |
|---|---|
| Sharpe | `annualized return / annualized volatility` |
| Sortino | `annualized return / downside volatility` |
| Calmar | `annualized return / abs(max drawdown)` |
| Hit Rate | `P(r_t > 0)` |
| t-Statistic | Test of `H₀: E[r] = 0`; `|t| > 2` is approximately significant for larger samples |
| Max Drawdown | `min_t { NAV(t) / max_{s≤t} NAV(s) - 1 }` |

The risk-free rate is assumed to be zero, which is acceptable for a market-neutral strategy prototype.

---

## Benchmarks

### Buy-and-Hold Benchmark

Simple equal-weight buy-and-hold benchmark:

- Every asset receives weight `1 / n`
- Positions are held until the end of the sample
- No transaction costs are applied

### Random Long/Short Benchmark

Random long/short portfolio with identical gross and net exposure:

- Selects `n_long + n_short` assets uniformly from the full universe
- Does not condition on signal values
- Weights are drawn from `Dirichlet(1)`, equivalent to a uniform distribution on the simplex per side
- Reuses the same selected `(L*, q*)` as the strategy to isolate weighting skill from parameter selection

---

## Project Structure

`src/`
- `data_loader.py` — yfinance download and monthly resampling
- `signals.py` — momentum, risk-adjusted momentum, composite signal
- `ic_analysis.py` — IC, ICIR, IC decay analysis
- `portfolio.py` — equal-weight and signal-weighted portfolio construction
- `backtest.py` — lagged-weight return computation and transaction costs
- `benchmark.py` — buy-and-hold benchmark
- `metrics.py` — Sharpe, Sortino, Calmar, hit rate, t-statistic, drawdown
- `random_portfolio.py` — random long/short benchmark

`main.py` — data loading, IC analysis, walk-forward backtest, plots

---

## Known Limitations

- Signals are currently weak in both return performance and IC analysis.
- Volatility weighting may reduce signal quality.
- Transaction cost model is simplified: flat 10 bps per unit turnover.
- No market impact or bid-ask spread model is included.
- Buy-and-hold benchmark is weak for comparison against long/short strategies.
- Random benchmark currently selects from the full universe instead of only the eligible top/bottom quantiles.
- Universe consists of roughly 90 current US large-cap stocks, introducing survivorship bias.
- Signal-weighted momentum can be sensitive to momentum crashes and reversal episodes.
- Higher net exposure can mechanically improve performance, creating market exposure bias.

---

## Planned Improvements

- Improve signal design.
- Add volatility-scaled position sizing.
- Add walk-forward IC analysis to detect signal decay over time.
- Restrict random benchmark selection to top/bottom signal quantiles.
- Improve transaction cost modeling.
- Add stronger benchmark models.
- Add additional signals.
- Improve reporting and visualization.
- Reduce or eliminate market exposure bias.
- Clean up plots and output formatting.

---

## Setup

Run the following commands:

`pip install pandas numpy scipy yfinance matplotlib`

`python main.py`