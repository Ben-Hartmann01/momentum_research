# r_pf(t) = Σ_i w_i(t-1)·r_i(t)   lagged wt → no look-ahead
def compute_returns(weights, returns):
    return (weights.shift(1) * returns).sum(axis=1)


# TO(t) = Σ_i |Δw_i(t)|,  cost = TO·rate
# 10bps flat: no mkt impact, no spread — conservative floor
def apply_transaction_costs(weights, strategy_returns, cost_rate=0.001):
    TO  = weights.diff().abs().sum(axis=1)
    return strategy_returns - TO * cost_rate, TO
