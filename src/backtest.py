# lagged weights: w(t-1) * r(t) — no look-ahead
def compute_returns(weights, returns):
    return (weights.shift(1) * returns).sum(axis=1)


# flat 10bps per unit turnover — crude floor, ignores spread and market impact
# TO(t) = Σ_i |w_i(t) - w_i(t-1)|
def apply_transaction_costs(weights, strategy_returns, cost_rate=0.001):
    TO = weights.diff().abs().sum(axis=1)
    return strategy_returns - TO * cost_rate, TO
