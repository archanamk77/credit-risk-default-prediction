"""Cleaning and feature engineering, shared by training and the Streamlit app."""
import pandas as pd

TARGET = "default"
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
BILLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYS = [f"PAY_AMT{i}" for i in range(1, 7)]
DELAYS = [f"PAY_{i}" for i in range(1, 7)]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Undocumented codes (0, 5, 6) are merged into 'others'
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["avg_bill"] = df[BILLS].mean(axis=1)
    df["utilization"] = (df["avg_bill"] / df["LIMIT_BAL"]).clip(-1, 3)
    df["pay_ratio"] = (df[PAYS].sum(axis=1) / (df[BILLS].clip(lower=0).sum(axis=1) + 1)).clip(0, 5)
    df["max_delay"] = df[DELAYS].max(axis=1)
    df["months_delayed"] = (df[DELAYS] > 0).sum(axis=1)
    return df


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    return add_features(clean(df))
