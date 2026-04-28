import yfinance as yf
import pandas as pd
import numpy as np



def get_price_data(tickers, start, end):
    """
    Download adjusted close prices from Yahoo Finance
    Returns a DataFrame with dates as index and tickers as columns
    """
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False
    )

    # Handle both single and multi-index columns
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" not in data.columns.get_level_values(0):
            raise ValueError("Adjusted close prices not found in downloaded data.")
        prices = data["Adj Close"].copy()
    else:
        if "Adj Close" not in data.columns:
            raise ValueError("Adjusted close prices not found in downloaded data.")
        prices = data[["Adj Close"]].copy()
        prices.columns = tickers

    prices = prices.sort_index()
    return prices

def clean_data(prices):
    """
    Clean price data by:
    - sorting index
    - removing duplicate dates
    - dropping rows with missing values
    """
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="first")]

    missing_summary = prices.isna().sum()
    print("Missing values by ticker before dropping rows:")
    print(missing_summary)

    cleaned = prices.dropna().copy()

    print("\nShape before dropna:", prices.shape)
    print("Shape after dropna: ", cleaned.shape)

    return cleaned

def compute_log_returns(prices):
    """
    Compute daily log returns from price data.
    """
    log_returns = np.log(prices / prices.shift(1))
    log_returns = log_returns.dropna().copy()
    return log_returns

def split_sample(returns, insample_end="2018-12-31"):
    """
    Split returns into in-sample and out-of-sample periods.
    """
    insample = returns.loc[:insample_end].copy()
    outsample = returns.loc[pd.Timestamp(insample_end) + pd.Timedelta(days=1):].copy()

    return insample, outsample

def compute_weighted_returns(ret_out: pd.DataFrame, state_series: pd.Series, w_0: pd.Series, w_1: pd.Series) -> pd.DataFrame:
    """
    Compute out-of-sample returns for regime-switching strategy using weights obtained from portfolio model.
    """
    asset_cols = ret_out.columns.tolist()

    w0 = w_0.loc[asset_cols].values
    w1 = w_1.loc[asset_cols].values

    results = []

    for date in ret_out.index:
        r_t = ret_out.loc[date, asset_cols].values
        state_t = state_series.loc[date]

        if state_t == 0:
            weight_vec = w0
        else:
            weight_vec = w1

        port_ret = np.dot(r_t, weight_vec)

        results.append({
            "date": date,
            "state": state_t,
            "portfolio_return": port_ret
        })

    result_df = pd.DataFrame(results).set_index("date")
    return result_df

def portfolio_performance_summary(return_series: pd.Series) -> pd.Series:
    """
    Compute performance summary statistics for a portfolio return series.
    """
    mean_daily = return_series.mean()
    vol_daily = return_series.std()

    sharpe = np.nan
    if vol_daily > 0:
        sharpe = (mean_daily * 252) / (vol_daily * np.sqrt(252))

    cumulative = np.exp(return_series.cumsum())
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    max_drawdown = drawdown.min()

    total_return = cumulative.iloc[-1] - 1

    return pd.Series({
        "mean_daily": mean_daily,
        "vol_daily": vol_daily,
        "annualized_return_approx": mean_daily * 252,
        "annualized_vol_approx": vol_daily * np.sqrt(252),
        "sharpe_approx": sharpe,
        "max_drawdown": max_drawdown,
        "total_return": total_return
    })
    
def compute_probability_weighted_returns(ret_out: pd.DataFrame, prob_df: pd.DataFrame, w_0: pd.Series, w_1: pd.Series) -> pd.DataFrame:
    """
    Compute portfolio returns using probability-weighted portfolios (focusing on out-of-sample).
    """
    asset_cols = ret_out.columns.tolist()

    w0 = w_0.loc[asset_cols].values
    w1 = w_1.loc[asset_cols].values

    results = []

    for date in ret_out.index:
        r_t = ret_out.loc[date, asset_cols].values

        p_low = prob_df.loc[date, "p_low_vol"]
        p_high = prob_df.loc[date, "p_high_vol"]

        # weighted portfolio
        w_t = p_low * w0 + p_high * w1

        port_ret = np.dot(r_t, w_t)

        results.append({
            "date": date,
            "portfolio_return": port_ret,
            "p_high_vol": p_high
        })

    result_df = pd.DataFrame(results).set_index("date")
    return result_df