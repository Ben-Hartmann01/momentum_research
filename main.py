from src.data_loader import load_data, get_monthly_prices, get_monthly_returns
from src.signals import momentum_signal, risk_adjusted_momentum, composite_signal
from src.portfolio import compute_weights_signal_weighted
from src.backtest import compute_returns, apply_transaction_costs
from src.metrics import (
    performance_metrics, drawdown,
    sortino_ratio, calmar_ratio, hit_rate, t_statistic,
)
from src.random_portfolio import compute_random_weights_signal_based
from src.ic_analysis import compute_ic, ic_summary, compute_ic_decay

import matplotlib.pyplot as plt
import pandas as pd

# ── Universe ───────────────────────────────────────────────────────────────────
tickers = [
    "AAPL", "MSFT", "AMZN", "META", "NVDA", "GOOGL", "GOOG", "TSLA",
    "ADBE", "CRM", "ORCL", "CSCO", "INTC", "AMD", "QCOM", "TXN", "AVGO",
    "JPM", "GS", "MS", "BAC", "WFC", "C", "BLK", "SCHW",
    "JNJ", "PFE", "MRK", "ABBV", "LLY", "TMO", "ABT", "DHR", "BMY", "GILD",
    "KO", "PEP", "WMT", "COST", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW",
    "XOM", "CVX", "COP", "SLB", "EOG", "PSX",
    "CAT", "BA", "GE", "HON", "UPS", "FDX", "DE", "LMT", "RTX", "MMM",
    "NEE", "DUK", "SO", "AEP", "EXC",
    "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS",
    "SPGI", "ICE", "ADP", "INTU", "ISRG", "MU", "PYPL", "AMAT", "KLAC", "LRCX",
]

start_date   = "2014-01-01"
end_date     = "2026-04-10"
lookbacks    = [3, 6, 9, 12]
quantiles    = [0.1, 0.2, 0.3, 0.4, 0.5]
train_window = 36   # months in-sample per fold
test_window  = 6    # months OOS per fold

# single source of truth — IC analysis & walk-forward iterate the same list
SIGNALS = [
    ("Raw Momentum",      momentum_signal),
    ("Risk-Adj Momentum", risk_adjusted_momentum),
    ("Composite Signal",  composite_signal),
]


# ── Param selection ────────────────────────────────────────────────────────────
# grid over (L, q): pick max in-sample Sharpe — normalises across vol regimes
def select_best_params(train_prices, train_returns, lookbacks, quantiles,
                       signal_func, weight_func):
    best_sharpe, best_params = None, None

    for lb in lookbacks:
        for q in quantiles:
            signals        = signal_func(train_prices, lb)
            weights        = weight_func(signals, long_quantile=q, short_quantile=q)
            rets           = compute_returns(weights, train_returns)
            rets_net, _    = apply_transaction_costs(weights, rets)
            _, _, sharpe   = performance_metrics(rets_net.dropna())

            if best_sharpe is None or sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = {"lookback": lb, "quantile": q}

    return best_params, best_sharpe


# ── Single OOS block ───────────────────────────────────────────────────────────
# signal on full [0:test_end]: rolling windows need history before test period
# eval on [test_start:test_end] only → no bleed from future prices
def run_test_block(prices, returns, test_start, test_end,
                   lookback, quantile, signal_func, weight_func):
    bp = prices.iloc[:test_end].copy()
    br = returns.iloc[:test_end].copy()

    signals      = signal_func(bp, lookback)
    weights      = weight_func(signals, long_quantile=quantile, short_quantile=quantile)
    tw           = weights.iloc[test_start:test_end].copy()
    tr           = br.iloc[test_start:test_end].copy()

    net_rets, _  = apply_transaction_costs(tw, compute_returns(tw, tr))
    return tw, net_rets


# ── Summary ────────────────────────────────────────────────────────────────────
def summarize_results(returns_series: pd.Series):
    cum   = (1 + returns_series.fillna(0)).cumprod()
    clean = returns_series.dropna()
    ann_ret, ann_vol, sharpe = performance_metrics(clean)
    dd, max_dd = drawdown(cum)

    return {
        "returns":    returns_series,
        "cumulative": cum,
        "ann_ret":    ann_ret,
        "ann_vol":    ann_vol,
        "sharpe":     sharpe,
        "sortino":    sortino_ratio(clean),
        "calmar":     calmar_ratio(clean, cum),
        "hit_rate":   hit_rate(clean),
        "t_stat":     t_statistic(clean),
        "drawdown":   dd,
        "max_dd":     max_dd,
    }


def print_summary(title, summary):
    t, p = summary["t_stat"]
    print(f"\n{title}")
    print(f"  ann ret  : {summary['ann_ret']:.4f}")
    print(f"  ann vol  : {summary['ann_vol']:.4f}")
    print(f"  Sharpe   : {summary['sharpe']:.4f}")
    print(f"  Sortino  : {summary['sortino']:.4f}")
    print(f"  Calmar   : {summary['calmar']:.4f}")
    print(f"  hit rate : {summary['hit_rate']*100:.1f}%")
    print(f"  max DD   : {summary['max_dd']:.4f}")
    print(f"  t-stat   : {t:.2f}  (p={p:.3f})")


# ── Walk-forward ───────────────────────────────────────────────────────────────
# train [t-36,t) → (L*,q*) → test [t,t+6) → roll +6
# same (L*,q*) for strat & BM: isolates weighting skill from param luck
def run_walk_forward(prices, returns, lookbacks, quantiles,
                     train_window, test_window, signal_func):
    strat_rets, bm_rets, param_log = [], [], []
    n = len(prices)

    for ts in range(train_window, n - test_window + 1, test_window):
        te = ts + test_window

        bp, br = prices.iloc[ts - train_window:ts].copy(), returns.iloc[ts - train_window:ts].copy()
        params, sharpe_is = select_best_params(
            bp, br, lookbacks, quantiles, signal_func, compute_weights_signal_weighted,
        )
        L, q = params["lookback"], params["quantile"]

        _, sr = run_test_block(prices, returns, ts, te, L, q,
                               signal_func, compute_weights_signal_weighted)
        _, br_ = run_test_block(prices, returns, ts, te, L, q,
                                signal_func, compute_random_weights_signal_based)

        strat_rets.append(sr)
        bm_rets.append(br_)
        param_log.append({"test_start": prices.index[ts], "test_end": prices.index[te-1],
                          "lookback": L, "quantile": q, "train_sharpe": sharpe_is})

    return {
        "strategy":  summarize_results(pd.concat(strat_rets).sort_index()),
        "benchmark": summarize_results(pd.concat(bm_rets).sort_index()),
        "params":    pd.DataFrame(param_log),
    }


# ── IC analysis ────────────────────────────────────────────────────────────────
# full-sample: no fitted params → no look-ahead
# forward_ret at t = returns.shift(-1).loc[t] = r_{t→t+1}
def run_ic_analysis(prices, returns, lookback=12):
    fwd = returns.shift(-1)

    print("\n" + "=" * 50)
    print("SIGNAL QUALITY  (IC / ICIR)")
    print("=" * 50)

    ic_all = {}
    for name, fn in SIGNALS:
        ic       = compute_ic(fn(prices, lookback), fwd)
        ic_all[name] = ic
        ic_summary(ic, name)

    print("\n--- IC Decay (cum ret, L=12) ---")
    decay_all = {}
    for name, fn in SIGNALS:
        d = compute_ic_decay(prices, fn, max_horizon=6, lookback=lookback)
        decay_all[name] = d
        print(f"  {name:<22}: " + "  ".join(f"h{h}={v:.3f}" for h, v in d.items()))

    return ic_all, decay_all


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    data    = load_data(tickers, start_date, end_date)
    prices  = get_monthly_prices(data)
    returns = get_monthly_returns(prices)

    returns = returns.dropna()
    prices  = prices.loc[returns.index]

    # 1. signal quality
    ic_all, decay_all = run_ic_analysis(prices, returns)

    # 2. walk-forward — same signal-wt construction for all signals
    kw = dict(prices=prices, returns=returns, lookbacks=lookbacks,
              quantiles=quantiles, train_window=train_window, test_window=test_window)

    results = {}
    for name, fn in SIGNALS:
        print(f"\nwalk-forward: {name} ...")
        results[name] = run_walk_forward(**kw, signal_func=fn)

    # 3. print
    print("\n" + "=" * 50)
    print("OOS RESULTS  (signal-weighted)")
    print("=" * 50)
    for name in results:
        print_summary(f"Strategy  — {name}", results[name]["strategy"])
    print()
    for name in results:
        print_summary(f"Benchmark — {name}", results[name]["benchmark"])

    for name in results:
        print(f"\nPARAMS — {name}")
        print(results[name]["params"].to_string(index=False))

    # ── Fig 1: IC over time + decay ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Signal IC Analysis", fontsize=13)

    ax = axes[0]
    for name, ic in ic_all.items():
        ax.plot(ic.rolling(12).mean(), label=f"{name} (12m MA)")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("Rolling 12m Mean IC")
    ax.set_ylabel("IC (Spearman)")
    ax.legend(); ax.grid(True)

    ax    = axes[1]
    pos   = list(next(iter(decay_all.values())).index)
    w     = 0.25
    for i, (name, d) in enumerate(decay_all.items()):
        ax.bar([p + (i-1)*w for p in pos], d.values, width=w, label=name)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("IC Decay  (cum ret, L=12)")
    ax.set_xlabel("horizon (months)")
    ax.set_ylabel("mean IC")
    ax.set_xticks(pos); ax.legend(); ax.grid(True, axis="y")

    plt.tight_layout(); plt.show()

    # ── Fig 2: cumulative returns ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    for (name, res), color in zip(results.items(), ["tab:blue", "tab:orange", "tab:green"]):
        ax.plot(res["strategy"]["cumulative"],  label=f"Strat — {name}", color=color, lw=2)
        ax.plot(res["benchmark"]["cumulative"], label=f"BM   — {name}", color=color, lw=1, ls=":")

    ax.set_title("Walk-Forward: Strategy vs BM  (signal-weighted)")
    ax.set_ylabel("cum ret  (1 = start)")
    ax.legend(); ax.grid(True)
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
