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


# ── Data params ────────────────────────────────────────────────────────────────

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
lookbacks             = [3, 6, 9, 12]        # months
quantiles             = [0.1, 0.2, 0.3, 0.4, 0.5]
target_gross_exposure = 2
train_window          = 36   # months for in-sample param selection
test_window           = 6    # months OOS per fold


# ── Central registries ─────────────────────────────────────────────────────────

# SIGNAL_REGISTRY: drives IC analysis — add a signal here to include it in IC output
# mean reversion pinned to 1m lookback regardless of the lb argument passed externally
SIGNAL_REGISTRY = {
    "Momentum":       momentum_signal,
    "Risk-Adj Mom":   risk_adjusted_momentum,
    "Mean Reversion": lambda p, lb: mean_reversion_signal(p, lookback=1),
    "Composite":      composite_signal,
}

# PORTFOLIO_CONFIGS: single source of truth for walk-forward, metrics table, and plots
# add/remove rows here to change what gets evaluated — nothing else needs touching
PORTFOLIO_CONFIGS = [
    {"name": "Momentum | equal | L/S",             "signal": "Momentum",  "method": "equal",  "net": 0.0},
    {"name": "Momentum | signal-wt | L/S",          "signal": "Momentum",  "method": "signal", "net": 0.0},
    {"name": "Momentum | equal | half-L/S",         "signal": "Momentum",  "method": "equal",  "net": 0.5},
    {"name": "Momentum | signal-wt | half-L/S",     "signal": "Momentum",  "method": "signal", "net": 0.5},
    {"name": "Momentum | equal | long-only",        "signal": "Momentum",  "method": "equal",  "net": 1.0},
    {"name": "Momentum | signal-wt | long-only",    "signal": "Momentum",  "method": "signal", "net": 1.0},
    {"name": "Composite | equal | L/S",             "signal": "Composite", "method": "equal",  "net": 0.0},
    {"name": "Composite | signal-wt | L/S",         "signal": "Composite", "method": "signal", "net": 0.0},
    {"name": "Composite | equal | half-L/S",        "signal": "Composite", "method": "equal",  "net": 0.5},
    {"name": "Composite | signal-wt | half-L/S",    "signal": "Composite", "method": "signal", "net": 0.5},
    {"name": "Composite | equal | long-only",       "signal": "Composite", "method": "equal",  "net": 1.0},
    {"name": "Composite | signal-wt | long-only",   "signal": "Composite", "method": "signal", "net": 1.0},
]

BENCHMARK_NAME = "Benchmark (EW L/O)"


# ── Weight functions ───────────────────────────────────────────────────────────

_WEIGHT_FUNCTIONS = {
    "equal":         compute_weights,
    "signal":        compute_weights_signal_weighted,
    "random_equal":  compute_random_weights,    # available but not in default configs
    "random_signal": compute_random_weights_signal_based,
}

def get_weight_function(method: str):
    if method not in _WEIGHT_FUNCTIONS:
        raise ValueError(f"Unknown method: {method!r}")
    return _WEIGHT_FUNCTIONS[method]


def compute_signal(prices, signal_key: str, lookback: int):
    if signal_key not in SIGNAL_REGISTRY:
        raise ValueError(f"Unknown signal: {signal_key!r}")
    return SIGNAL_REGISTRY[signal_key](prices, lookback)


# ── Param selection ────────────────────────────────────────────────────────────

# grid over (lookback, quantile) — pick combo with max in-sample Sharpe
# Sharpe normalizes across vol regimes better than raw return
def select_best_params(train_prices, train_returns, lookbacks, quantiles,
                       method="equal", signal_key="Momentum",
                       target_net_exposure=0.0, target_gross_exposure=2.0):
    best_sharpe = None
    best_params = None
    weight_func = get_weight_function(method)

    for lb in lookbacks:
        for q in quantiles:
            signals = compute_signal(train_prices, signal_key, lb)
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

# signal computed on [0, test_end] so rolling lookbacks have full warmup history
# only the [test_start, test_end] slice is actually used for evaluation
def run_test_block(prices, returns, test_start, test_end, lookback, quantile,
                   method="equal", signal_key="Momentum",
                   target_net_exposure=0.0, target_gross_exposure=2.0):
    block_prices  = prices.iloc[:test_end].copy()
    block_returns = returns.iloc[:test_end].copy()

    signals     = compute_signal(block_prices, signal_key, lookback)
    weight_func = get_weight_function(method)

    weights = weight_func(
        signals,
        long_quantile=quantile,
        short_quantile=quantile,
        target_net_exposure=target_net_exposure,
        target_gross_exposure=target_gross_exposure,
    )

    test_weights = weights.iloc[test_start:test_end].copy()
    test_returns = block_returns.iloc[test_start:test_end].copy()
    net_rets, turnover = apply_transaction_costs(
        test_weights, compute_returns(test_weights, test_returns)
    )
    return test_weights, net_rets, turnover


# ── Benchmark block ────────────────────────────────────────────────────────────

def run_equal_weight_benchmark_block(prices, returns, test_start, test_end):
    block_prices  = prices.iloc[:test_end].copy()
    block_returns = returns.iloc[:test_end].copy()

    # notna mask as signal — every live asset gets equal weight
    bm_signals = block_prices.notna().astype(float)
    bm_weights = compute_equal_weight_long_only_weights(bm_signals)

    test_weights = bm_weights.iloc[test_start:test_end].copy()
    test_returns = block_returns.iloc[test_start:test_end].copy()

    bm_returns, turnover = apply_transaction_costs(
        test_weights, compute_returns(test_weights, test_returns)
    )
    return test_weights, bm_returns, turnover


# ── Summary ────────────────────────────────────────────────────────────────────

def summarize_results(returns_series: pd.Series) -> dict:
    cum   = (1 + returns_series.fillna(0)).cumprod()  # compounding NAV
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


# ── Walk-forward (strategy) ────────────────────────────────────────────────────

# train [t-36, t) → select (lb*, q*) → test [t, t+6) → shift +6, repeat
def run_walk_forward(prices, returns, lookbacks, quantiles, train_window, test_window,
                     method="equal", signal_key="Momentum",
                     target_net_exposure=0.0, target_gross_exposure=2.0):
    all_returns     = []
    selected_params = []
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
            signal_key=signal_key,
            target_net_exposure=target_net_exposure,
            target_gross_exposure=target_gross_exposure,
        )

        _, rets_net, _ = run_test_block(
            prices=prices,
            returns=returns,
            test_start=test_start,
            test_end=test_end,
            lookback=best_params["lookback"],
            quantile=best_params["quantile"],
            method=method,
            signal_key=signal_key,
            target_net_exposure=target_net_exposure,
            target_gross_exposure=target_gross_exposure,
        )

        all_returns.append(rets_net)
        selected_params.append({
            "test_start":   prices.index[test_start],
            "test_end":     prices.index[test_end - 1],
            "lookback":     best_params["lookback"],
            "quantile":     best_params["quantile"],
            "train_sharpe": best_sharpe,
        })

    return {
        "strategy": summarize_results(pd.concat(all_returns).sort_index()),
        "params":   pd.DataFrame(selected_params),
    }


# ── Walk-forward (benchmark) ───────────────────────────────────────────────────

# same fold structure as the strategies — computed once, reused for all comparisons
def run_benchmark_walk_forward(prices, returns, train_window, test_window):
    all_returns = []
    n = len(prices)
    for test_start in range(train_window, n - test_window + 1, test_window):
        test_end = test_start + test_window
        _, rets, _ = run_equal_weight_benchmark_block(prices, returns, test_start, test_end)
        all_returns.append(rets)
    return summarize_results(pd.concat(all_returns).sort_index())


# ── IC analysis ────────────────────────────────────────────────────────────────

# full-sample, no fitted params → no look-ahead bias here
# fwd at t = returns.shift(-1).loc[t] = r_{t→t+1}
def run_ic_analysis(prices, returns, lookback=12):
    fwd = returns.shift(-1)

    print("\n" + "=" * 60)
    print("SIGNAL QUALITY  (IC / ICIR)")
    print("=" * 60)

    ic_all    = {}
    decay_all = {}

    for name, fn in SIGNAL_REGISTRY.items():
        ic           = compute_ic(fn(prices, lookback), fwd)
        ic_all[name] = ic
        ic_summary(ic, name)

    print("\n--- IC Decay (cumulative return, L=12) ---")
    for name, fn in SIGNAL_REGISTRY.items():
        d               = compute_ic_decay(prices, fn, max_horizon=6, lookback=lookback)
        decay_all[name] = d
        print(f"  {name:<22}: " + "  ".join(f"h{h}={v:.3f}" for h, v in d.items()))

    return ic_all, decay_all


# ── Metrics table ──────────────────────────────────────────────────────────────

def build_metrics_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, s in results.items():
        t, p = s["t_stat"]
        rows.append({
            "Ann Ret":  s["ann_ret"],
            "Ann Vol":  s["ann_vol"],
            "Sharpe":   s["sharpe"],
            "Sortino":  s["sortino"],
            "Calmar":   s["calmar"],
            "Hit%":     s["hit_rate"] * 100,
            "Max DD":   s["max_dd"],
            "t-stat":   t,
            "p-val":    p,
        })
    df = pd.DataFrame(rows, index=pd.Index(list(results.keys()), name="Strategy"))
    return df.round(4)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    data    = load_data(tickers, start_date, end_date)
    prices  = get_monthly_prices(data)
    returns = get_monthly_returns(prices)

    # drop leading NaNs — first row is always NaN from pct_change
    returns = returns.dropna()
    prices  = prices.loc[returns.index]

    # 1. signal quality — all signals in SIGNAL_REGISTRY, full sample, no params fitted
    ic_all, decay_all = run_ic_analysis(prices, returns)

    # 2. benchmark — computed once here, shown alongside all strategies below
    benchmark = run_benchmark_walk_forward(prices, returns, train_window, test_window)

    # 3. walk-forward for every entry in PORTFOLIO_CONFIGS — nothing else
    results = {}
    for cfg in PORTFOLIO_CONFIGS:
        wf = run_walk_forward(
            prices=prices,
            returns=returns,
            lookbacks=lookbacks,
            quantiles=quantiles,
            train_window=train_window,
            test_window=test_window,
            method=cfg["method"],
            signal_key=cfg["signal"],
            target_net_exposure=cfg["net"],
            target_gross_exposure=target_gross_exposure,
        )
        results[cfg["name"]] = wf["strategy"]

    results[BENCHMARK_NAME] = benchmark  # append last so it's at the bottom of the table

    # 4. metrics table — same results dict, same order
    print("\n" + "=" * 60)
    print("WALK-FORWARD PERFORMANCE SUMMARY")
    print("=" * 60)
    tbl = build_metrics_table(results)
    print(tbl.to_string())

    # 5. IC analysis plots — same SIGNAL_REGISTRY as step 1
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
    n_sig = len(decay_all)
    w     = 0.18
    for i, (name, d) in enumerate(decay_all.items()):
        offset = (i - (n_sig - 1) / 2) * w  # center bars around each horizon tick
        ax.bar([p + offset for p in pos], d.values, width=w, label=name)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title("IC Decay  (cum ret, L=12)")
    ax.set_xlabel("horizon (months)")
    ax.set_ylabel("mean IC")
    ax.set_xticks(pos); ax.legend(); ax.grid(True, axis="y")

    plt.tight_layout(); plt.show()

    # 6. cumulative returns — same PORTFOLIO_CONFIGS as step 3
    # split into two panels so each is readable (6 lines + benchmark each)
    equal_cfgs  = [c for c in PORTFOLIO_CONFIGS if c["method"] == "equal"]
    signal_cfgs = [c for c in PORTFOLIO_CONFIGS if c["method"] == "signal"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle("Walk-Forward Cumulative Returns", fontsize=13)

    for ax, cfgs, title in [
        (axes[0], equal_cfgs,  "Equal-Weighted"),
        (axes[1], signal_cfgs, "Signal-Weighted"),
    ]:
        for cfg in cfgs:
            label = f"{cfg['signal']} | net={cfg['net']}"
            ax.plot(results[cfg["name"]]["cumulative"], label=label)
        ax.plot(benchmark["cumulative"], color="black", lw=1.5, ls="--", label=BENCHMARK_NAME)
        ax.set_title(title)
        ax.set_ylabel("Cumulative return")
        ax.legend(fontsize=8); ax.grid(True)

    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
