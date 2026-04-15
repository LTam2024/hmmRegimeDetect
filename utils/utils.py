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