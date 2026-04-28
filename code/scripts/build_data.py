import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import *

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)


# Change these parameters as needed depending on what periods/assets to focus on
tickers = ["SPY", "QQQ", "IWM", "EFA", "TLT", "LQD", "GLD", "HYG"]
marketTicker = "SPY"

start = "2007-01-01"
end = "2025-12-31"
insampleEnd = "2018-12-31"

def build_dataset():
    prices = get_price_data(tickers, start, end)
    prices = clean_data(prices)

    returns = compute_log_returns(prices)

    # regime detection series
    regime_returns = returns[[marketTicker]].copy()

    # split full asset returns
    ret_in, ret_out = split_sample(returns, insample_end=insampleEnd)

    # split SPY-only regime returns
    regime_in, regime_out = split_sample(regime_returns, insample_end=insampleEnd)

    print("\nFinal dataset summary")
    print("---------------------")
    print("Prices shape:", prices.shape)
    print("Returns shape:", returns.shape)
    print("In-sample returns shape:", ret_in.shape)
    print("Out-of-sample returns shape:", ret_out.shape)

    return {
        "prices": prices,
        "returns": returns,
        "regime_returns": regime_returns,
        "ret_in": ret_in,
        "ret_out": ret_out,
        "regime_in": regime_in,
        "regime_out": regime_out,
    }
if __name__ == "__main__":
    data = build_dataset()

    #Save outputs to CSV files for use in later portions of project
    data["prices"].to_csv(os.path.join(DATA_DIR, "etf_prices.csv"))
    data["returns"].to_csv(os.path.join(DATA_DIR, "etf_returns.csv"))
    data["ret_in"].to_csv(os.path.join(DATA_DIR, "etf_returns_insample.csv"))
    data["ret_out"].to_csv(os.path.join(DATA_DIR, "etf_returns_outsample.csv"))