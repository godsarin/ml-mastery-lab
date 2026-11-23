"""Feature engineering and feature selection helpers for the Telco churn project.

All functions are rule-based: they look at one row at a time and learn nothing from the
data, so they are safe to apply to training data, test data and brand-new customers alike
(no data leakage).
"""
import pandas as pd

# The six optional "add-on" services a customer can have
ADDON_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# Features dropped after feature selection (notebook 07)
DROP_COLS = ["gender", "PhoneService", "MultipleLines"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with three new columns."""
    out = df.copy()

    # How many add-on services does the customer pay for?
    out["num_addons"] = (out[ADDON_COLS] == "Yes").sum(axis=1)

    # Does the customer pay automatically (bank transfer or credit card)?
    out["is_autopay"] = out["PaymentMethod"].str.contains("automatic").astype(int)

    # Group tenure into stages of the customer relationship
    out["tenure_group"] = pd.cut(
        out["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"],
    ).astype(str)

    return out


def drop_weak_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df without the columns chosen for removal."""
    return df.drop(columns=DROP_COLS)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Full step used by the pipeline: add new features, then drop weak ones."""
    return drop_weak_features(add_features(df))
