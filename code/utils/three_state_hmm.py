from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import pandas as pd


def fit_three_state_hmm(regime_in: pd.DataFrame, n_states: int = 3):
    """
    Fit a Gaussian HMM to a 1-column return DataFrame.
    Named "three_state_hmm" for my use case, but can be used with any n_states.
    """
    if regime_in.shape[1] != 1:
        raise ValueError("regime_in must have exactly one column.")

    scaler = StandardScaler()
    X = scaler.fit_transform(regime_in.values)
    
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=1000,
        tol=1e-5,
        random_state=42
    )
    model.fit(X)

    states = model.predict(X)
    probs = model.predict_proba(X)

    state_df = regime_in.copy()
    state_df.columns = ["SPY_ret"]
    state_df["state_raw"] = states

    for j in range(n_states):
        state_df[f"p_state_{j}"] = probs[:, j]

    return model, state_df, scaler

def relabel_three_states_by_vol(state_df: pd.DataFrame):
    """
    Relabel 3 raw states by increasing volatility:
    0 = low-vol, 1 = medium-vol, 2 = high-vol
    """
    vols = state_df.groupby("state_raw")["SPY_ret"].std().sort_values()

    ordered_raw_states = vols.index.tolist()  
    mapping = {
        ordered_raw_states[0]: 0,
        ordered_raw_states[1]: 1,
        ordered_raw_states[2]: 2
    }

    relabeled = state_df.copy()
    relabeled["state"] = relabeled["state_raw"].map(mapping)

    # map probability
    for raw_state, new_state in mapping.items():
        relabeled[f"p_state_relab_{new_state}"] = relabeled[f"p_state_{raw_state}"]

    relabeled["p_low_vol"] = relabeled["p_state_relab_0"]
    relabeled["p_mid_vol"] = relabeled["p_state_relab_1"]
    relabeled["p_high_vol"] = relabeled["p_state_relab_2"]

    return relabeled, mapping

def classify_outsample_regimes_nstate(model, scaler, regime_out: pd.DataFrame, n_states: int):
    """
    
    Classify out-of-sample returns into HMM states using the fitted model and scaler.
    Can be used for any n_states, but currently only have 2 or 3 state use cases.
    """
    X_out = scaler.transform(regime_out.values)
    states_out = model.predict(X_out)
    probs_out = model.predict_proba(X_out)

    out_df = regime_out.copy()
    out_df.columns = ["SPY_ret"]
    out_df["state_raw"] = states_out

    for j in range(n_states):
        out_df[f"p_state_{j}"] = probs_out[:, j]

    return out_df


def apply_three_state_mapping(out_df: pd.DataFrame, mapping: dict):
    relabeled = out_df.copy()
    relabeled["state"] = relabeled["state_raw"].map(mapping)

    for raw_state, new_state in mapping.items():
        relabeled[f"p_state_relab_{new_state}"] = relabeled[f"p_state_{raw_state}"]

    relabeled["p_low_vol"] = relabeled["p_state_relab_0"]
    relabeled["p_mid_vol"] = relabeled["p_state_relab_1"]
    relabeled["p_high_vol"] = relabeled["p_state_relab_2"]

    return relabeled