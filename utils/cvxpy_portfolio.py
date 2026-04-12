import numpy as np
import pandas as pd
import cvxpy as cp

def solve_min_variance_portfolio(cov_matrix: pd.DataFrame) -> pd.Series:
    """
    Solve minimum variance portfolio optimization problem using cvxpy with GUROBI solver.
    Returns a Series of portfolio weights.
    """
    n = cov_matrix.shape[0]
    w = cp.Variable(n)

    sigma = cov_matrix.values

    objective = cp.Minimize(cp.quad_form(w, sigma))
    constraints = [
        cp.sum(w) == 1,
        w >= 0
    ]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GUROBI)

    if w.value is None:
        raise ValueError("Optimization failed.")

    weights = pd.Series(w.value, index=cov_matrix.index)
    return weights

def portfolio_stats(weights: pd.Series, returns: pd.DataFrame) -> dict:
    """
    Compute portfolio statistics and return them.
    """
    port_ret = returns @ weights
    mean_daily = port_ret.mean()
    vol_daily = port_ret.std()

    return {
        "mean_daily": mean_daily,
        "vol_daily": vol_daily,
        "mean_annual_approx": mean_daily * 252,
        "vol_annual_approx": vol_daily * np.sqrt(252),
        "sharpe_approx": (mean_daily * 252) / (vol_daily * np.sqrt(252)) if vol_daily > 0 else np.nan
    }