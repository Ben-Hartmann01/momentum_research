from src.data_loader import load_data, get_monthly_prices, get_monthly_returns
from src.signals import momentum_signal, risk_adjusted_momentum, composite_signal, mean_reversion_signal
from src.portfolio import compute_weights, compute_weights_signal_weighted
from src.backtest import compute_returns, apply_transaction_costs
from src.metrics import (
    performance_metrics, drawdown,
    sortino_ratio, calmar_ratio, hit_rate, t_statistic,
)
from src.random_portfolio import compute_random_weights, compute_random_weights_signal_based
from src.benchmark import compute_equal_weight_long_only_weights
from src.ic_analysis import compute_ic, ic_summary, compute_ic_decay

import matplotlib.pyplot as plt
import pandas as pd


# ── Universe ───────────────────────────────────────────────────────────────────

def compute_signal(prices, signal_type, lookback):
    if signal_type == "momentum":
        return momentum_signal(prices, lookback=lookback)
    elif signal_type == "mean_reversion":
        return mean_reversion_signal(prices, lookback=lookback)
    elif signal_type == "composite":
        return composite_signal(prices, momentum_lookback=lookback)
    else:
        raise ValueError(f"Unknown signal_type: {signal_type}")

# Data

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

start_date            = "2014-01-01"
end_date              = "2026-04-10"
lookbacks             = [3, 6, 9, 12]
quantiles             = [0.1, 0.2, 0.3, 0.4, 0.5]
net_exposures         = [0.0, 0.5, 1.0]
target_gross_exposure = 2
train_window          = 36
test_window           = 6

# single source of truth — IC analysis & walk-forward iterate the same list
SIGNALS = [
    ("Raw Momentum",      momentum_signal),
    ("Risk-Adj Momentum", risk_adjusted_momentum),
    ("Composite Signal",  composite_signal),
]


# ── Weight functions ───────────────────────────────────────────────────────────

_WEIGHT_FUNCTIONS = {
    "equal":         compute_weights,
    "signal":        compute_weights_signal_weighted,
    "random_equal":  compute_random_weights,
    "random_signal": compute_random_weights_signal_based,
}

def get_weight_function(method: str):
    if method not in _WEIGHT_FUNCTIONS:
        raise ValueError(f"Unknown method: {method}")
    return _WEIGHT_FUNCTIONS[method]


# ── Param selection ────────────────────────────────────────────────────────────
# grid over (L, q): pick max in-sample Sharpe — normalises across vol regimes
def select_best_params(train_prices, train_returns, lookbacks, quantiles,
                       method="equal", signal_type="momentum",
                       target_net_exposure=0.0, target_gross_exposure=2.0):
    best_sharpe = None
    best_params = None

    weight_func = get_weight_function(method=method)

    for lb in lookbacks:
        for q in quantiles:
            signals = compute_signal(train_prices, signal_type=signal_type, lookback=lb)

            weights = weight_func(
                signals,
                long_quantile=q,
                short_quantile=q,
                target_net_exposure=target_net_exposure,
                target_gross_exposure=target_gross_exposure,
            )

            rets = compute_returns(weights, train_returns)
            rets_net, _ = apply_transaction_costs(weights, rets)

            _, _, sharpe = performance_metrics(rets_net.dropna())

            if best_sharpe is None or sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = {"lookback": lb, "quantile": q}

    return best_params, best_sharpe


# ── Single OOS block ───────────────────────────────────────────────────────────
# signal on full [0:test_end]: rolling windows need history before test period
# eval on [test_start:test_end] only → no bleed from future prices
def run_test_block(prices, returns, test_start, test_end, lookback, quantile,
                   method="equal", signal_type="momentum",
                   target_net_exposure=0.0, target_gross_exposure=2.0):
    block_prices  = prices.iloc[:test_end].copy()
    block_returns = returns.iloc[:test_end].copy()

    signals     = compute_signal(block_prices, signal_type=signal_type, lookback=lookback)
    weight_func = get_weight_function(method=method)

    weights = weight_func(
        signals,
        long_quantile=quantile,
        short_quantile=quantile,
        target_net_exposure=target_net_exposure,
        target_gross_exposure=target_gross_exposure,
    )

    test_weights  = weights.iloc[test_start:test_end].copy()
    test_returns  = block_returns.iloc[test_start:test_end].copy()

    net_rets, turnover = apply_transaction_costs(test_weights, compute_returns(test_weights, test_returns))
    return test_weights, net_rets, turnover


# ── Equal-weight benchmark block ───────────────────────────────────────────────

# New BM
def run_equal_weight_benchmark_block(prices, returns, test_start, test_end):
    block_prices  = prices.iloc[:test_end].copy()
    block_returns = returns.iloc[:test_end].copy()

    benchmark_signals = block_prices.notna().astype(float)
    benchmark_weights = compute_equal_weight_long_only_weights(benchmark_signals)

    test_weights  = benchmark_weights.iloc[test_start:test_end].copy()
    test_returns  = block_returns.iloc[test_start:test_end].copy()

    benchmark_returns = compute_returns(test_weights, test_returns)
    benchmark_returns_net, turnover = apply_transaction_costs(test_weights, benchmark_returns)

    return test_weights, benchmark_returns_net, turnover


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
def run_walk_forward(prices, returns, lookbacks, quantiles, train_window, test_window,
                     method="equal", signal_type="momentum",
                     target_net_exposure=0.0, target_gross_exposure=2.0):

    all_strategy_returns    = []
    all_benchmark_returns   = []
    selected_params_history = []

    n = len(prices)

    for test_start in range(train_window, n - test_window + 1, test_window):
        test_end = test_start + test_window

        train_prices  = prices.iloc[test_start - train_window:test_start].copy()
        train_returns = returns.iloc[test_start - train_window:test_start].copy()

        best_params, best_sharpe = select_best_params(
            train_prices=train_prices,
            train_returns=train_returns,
            lookbacks=lookbacks,
            quantiles=quantiles,
            method=method,
            signal_type=signal_type,
            target_net_exposure=target_net_exposure,
            target_gross_exposure=target_gross_exposure,
        )

        best_lb = best_params["lookback"]
        best_q  = best_params["quantile"]

        _, strategy_returns_net, _ = run_test_block(
            prices=prices,
            returns=returns,
            test_start=test_start,
            test_end=test_end,
            lookback=best_lb,
            quantile=best_q,
            method=method,
            signal_type=signal_type,
            target_net_exposure=target_net_exposure,
            target_gross_exposure=target_gross_exposure,
        )

        _, benchmark_returns_net, _ = run_equal_weight_benchmark_block(
            prices=prices,
            returns=returns,
            test_start=test_start,
            test_end=test_end,
        )

        all_strategy_returns.append(strategy_returns_net)
        all_benchmark_returns.append(benchmark_returns_net)
        selected_params_history.append({
            "test_start":   prices.index[test_start],
            "test_end":     prices.index[test_end - 1],
            "lookback":     best_lb,
            "quantile":     best_q,
            "train_sharpe": best_sharpe,
        })

    return {
        "strategy":  summarize_results(pd.concat(all_strategy_returns).sort_index()),
        "benchmark": summarize_results(pd.concat(all_benchmark_returns).sort_index()),
        "params":    pd.DataFrame(selected_params_history),
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
        ic           = compute_ic(fn(prices, lookback), fwd)
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

    # 2. walk-forward
    signal_types   = ["momentum", "composite"]
    equal_results  = {}
    signal_results = {}

    for sig in signal_types:
        for net in net_exposures:
            equal_results[(sig, net)] = run_walk_forward(
                prices=prices,
                returns=returns,
                lookbacks=lookbacks,
                quantiles=quantiles,
                train_window=train_window,
                test_window=test_window,
                method="equal",
                signal_type=sig,
                target_net_exposure=net,
                target_gross_exposure=target_gross_exposure,
            )

            signal_results[(sig, net)] = run_walk_forward(
                prices=prices,
                returns=returns,
                lookbacks=lookbacks,
                quantiles=quantiles,
                train_window=train_window,
                test_window=test_window,
                method="signal",
                signal_type=sig,
                target_net_exposure=net,
                target_gross_exposure=target_gross_exposure,
            )

    benchmark_results = equal_results[("momentum", 1.0)]["benchmark"]

    for (sig, net), results in equal_results.items():
        print_summary(f"Equal-weighted | Signal={sig} | Net={net}", results["strategy"])

    for (sig, net), results in signal_results.items():
        print_summary(f"Signal-weighted | Signal={sig} | Net={net}", results["strategy"])

    print_summary("BENCHMARK RESULTS (EQUAL-WEIGHT LONG-ONLY)", benchmark_results)

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
    plt.figure(figsize=(10, 6))

    for (sig, net), results in signal_results.items():
        plt.plot(
            results["strategy"]["cumulative"],
            label=f"{sig.capitalize()} (equal, net={net})"
        )

    plt.plot(benchmark_results["cumulative"], label="Benchmark (Equal-weight Long-only)")

    plt.legend()
    plt.title("Walk-Forward Strategies by Net Exposure")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
