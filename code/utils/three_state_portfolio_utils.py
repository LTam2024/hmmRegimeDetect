import pandas as pd
import numpy as np

def compute_three_state_weighted_returns(ret_out: pd.DataFrame, state_series: pd.Series, w_0: pd.Series, w_1: pd.Series, w_2: pd.Series) -> pd.DataFrame:
    """
    Compute returns for 3-state regime-switching strategy using weights obtained from portfolio model.
    """
    asset_cols = ret_out.columns.tolist()
    w0 = w_0.loc[asset_cols].values
    w1 = w_1.loc[asset_cols].values
    w2 = w_2.loc[asset_cols].values

    results = []

    for date in ret_out.index:
        r_t = ret_out.loc[date, asset_cols].values
        s_t = state_series.loc[date]

        if s_t == 0:
            w_t = w0
        elif s_t == 1:
            w_t = w1
        else:
            w_t = w2

        port_ret = np.dot(r_t, w_t)

        results.append({
            "date": date,
            "state": s_t,
            "portfolio_return": port_ret
        })

    return pd.DataFrame(results).set_index("date")

def compute_three_state_probability_weighted_returns(ret_out: pd.DataFrame, prob_df: pd.DataFrame, w_0: pd.Series, w_1: pd.Series, w_2: pd.Series) -> pd.DataFrame:
    """
    Compute returns for 3-state regime-switching strategy using probability-weighted combination of weights from portfolio model.
    """
    asset_cols = ret_out.columns.tolist()
    w0 = w_0.loc[asset_cols].values
    w1 = w_1.loc[asset_cols].values
    w2 = w_2.loc[asset_cols].values

    results = []

    for date in ret_out.index:
        r_t = ret_out.loc[date, asset_cols].values

        p0 = prob_df.loc[date, "p_low_vol"]
        p1 = prob_df.loc[date, "p_mid_vol"]
        p2 = prob_df.loc[date, "p_high_vol"]

        w_t = p0 * w0 + p1 * w1 + p2 * w2
        port_ret = np.dot(r_t, w_t)

        results.append({
            "date": date,
            "portfolio_return": port_ret,
            "p_high_vol": p2
        })

    return pd.DataFrame(results).set_index("date")