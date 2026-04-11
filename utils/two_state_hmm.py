"""
This module fits a 2-state Gaussian HMM to SPY in-sample returns and summarizes the results.

It uses random_state=42 for reproducibility. The states are relabeled so that:
- State 0 = lower-volatility regime
- State 1 = higher-volatility regime
The module also generates diagnostic plots and saves the state assignments.
Plots are saved to specified output directory, for purposes of this project will be in "results" directory.

"""
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



def fit_two_state_hmm(regime_in: pd.DataFrame) -> tuple[GaussianHMM, pd.DataFrame]:
    """
    Fit a 2-state Gaussian HMM to SPY in-sample returns.

    Parameters
    ----------
    regime_in : pd.DataFrame
        DataFrame with one column: SPY returns.

    Returns
    -------
    model : GaussianHMM
        Fitted HMM model.
    state_df : pd.DataFrame
        DataFrame with returns, most likely state, and smoothed probabilities.
    """
    if regime_in.shape[1] != 1:
        raise ValueError("regime_in should have exactly one column for SPY returns.")

    # hmmlearn expects 2D array: (n_samples, n_features)
    X = regime_in.values

    model = GaussianHMM(
        n_components=2,
        covariance_type="full",
        n_iter=500,
        tol=1e-4,
        random_state=42
    )
    model.fit(X)

    # Most likely state sequence
    states = model.predict(X)

    # Posterior state probabilities
    probs = model.predict_proba(X)

    state_df = regime_in.copy()
    state_df.columns = ["SPY_ret"]
    state_df["state"] = states
    state_df["p_state_0"] = probs[:, 0]
    state_df["p_state_1"] = probs[:, 1]

    return model, state_df


def fit_two_state_hmm_scaled(regime_in: pd.DataFrame) -> tuple[GaussianHMM, pd.DataFrame]:
    """
    Fit a 2-state Gaussian HMM to SPY in-sample returns using scaled data.

    Parameters
    ----------
    regime_in : pd.DataFrame
        DataFrame with one column: SPY returns.

    Returns
    -------
    model : GaussianHMM
        Fitted HMM model.
    state_df : pd.DataFrame
        DataFrame with returns, most likely state, and smoothed probabilities.
    """
    if regime_in.shape[1] != 1:
        raise ValueError("regime_in should have exactly one column for SPY returns.")

    scaler = StandardScaler()
    X = scaler.fit_transform(regime_in.values)

    model = GaussianHMM(
        n_components=2,
        covariance_type="full",
        n_iter=500,
        tol=1e-4,
        random_state=42
    )
    model.fit(X)

    # Most likely state sequence
    states = model.predict(X)

    # Posterior state probabilities
    probs = model.predict_proba(X)

    state_df = regime_in.copy()
    state_df.columns = ["SPY_ret"]
    state_df["state"] = states
    state_df["p_state_0"] = probs[:, 0]
    state_df["p_state_1"] = probs[:, 1]

    return model, state_df


def summarize_states(model: GaussianHMM, state_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize each state's mean, volatility, and frequency.
    """
    summary_rows = []

    for s in sorted(state_df["state"].unique()):
        subset = state_df[state_df["state"] == s]["SPY_ret"]

        summary_rows.append({
            "state": s,
            "n_obs": len(subset),
            "fraction": len(subset) / len(state_df),
            "mean_return": subset.mean(),
            "volatility": subset.std(),
            "annualized_mean_approx": subset.mean() * 252,
            "annualized_vol_approx": subset.std() * np.sqrt(252),
        })

    summary = pd.DataFrame(summary_rows).sort_values("state").reset_index(drop=True)
    return summary


def relabel_states_by_vol_two_state(state_df: pd.DataFrame) -> pd.DataFrame:
    """
    Relabel states so that:
    0 = lower-vol regime
    1 = higher-vol regime
    """
    vols = state_df.groupby("state")["SPY_ret"].std().sort_values()
    low_vol_state = vols.index[0]
    high_vol_state = vols.index[1]

    mapping = {
        low_vol_state: 0,
        high_vol_state: 1
    }

    relabeled = state_df.copy()
    relabeled["state"] = relabeled["state"].map(mapping)

    # remap probability columns
    if low_vol_state == 0 and high_vol_state == 1:
        relabeled["p_low_vol"] = relabeled["p_state_0"]
        relabeled["p_high_vol"] = relabeled["p_state_1"]
    else:
        relabeled["p_low_vol"] = relabeled["p_state_1"]
        relabeled["p_high_vol"] = relabeled["p_state_0"]

    return relabeled


def plot_regimes(state_df: pd.DataFrame, output_dir: str, plot_id: str) -> None:
    """
    Make basic diagnostic plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: returns colored by state
    plt.figure(figsize=(14, 5))
    for s in [0, 1]:
        mask = state_df["state"] == s
        plt.scatter(
            state_df.index[mask],
            state_df.loc[mask, "SPY_ret"],
            s=6,
            label=f"State {s}"
        )
    plt.title("SPY Daily Returns by HMM State")
    plt.ylabel("Daily log return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"spy_returns_by_state_{plot_id}.png"), dpi=200)
    plt.close()

    # Plot 2: high-vol state probability
    plt.figure(figsize=(14, 4))
    plt.plot(state_df.index, state_df["p_high_vol"])
    plt.title("Probability of High-Volatility Regime")
    plt.ylabel("Probability")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"high_vol_probability_{plot_id}.png"), dpi=200)
    plt.close()
    
