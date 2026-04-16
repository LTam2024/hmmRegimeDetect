import pandas as pd
import numpy as np

def equal_weight_returns(ret_out: pd.DataFrame) -> pd.Series:
    """
    Compute equal-weighted portfolio returns for returns (focusing on out-of-sample period).
    
    Returns
    -------
    pd.Series
        Series of equal-weighted portfolio returns.
    """
    n = ret_out.shape[1]
    w_eq = np.repeat(1 / n, n)
    return pd.Series(ret_out.values @ w_eq, index=ret_out.index, name="equal_weight")

# This function is usually used with a base mean-variance portfolio computed from in-sample returns
def fixed_weight_returns(ret_out: pd.DataFrame, weights: pd.Series, name="static_portfolio") -> pd.Series:
    """
    Compute fixed-weighted portfolio returns for returns (focusing on out-of-sample period).
    
    Returns
    -------
    pd.Series
        Series of fixed-weighted portfolio returns.
    """
    w = weights.loc[ret_out.columns].values
    return pd.Series(ret_out.values @ w, index=ret_out.index, name=name)