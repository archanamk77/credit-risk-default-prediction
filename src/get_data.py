"""Download the UCI 'Default of Credit Card Clients' dataset (30,000 rows, no login needed)."""
from pathlib import Path
from ucimlrepo import fetch_ucirepo

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "credit_default.csv"
COLS = (["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
        + ["PAY_1"] + [f"PAY_{i}" for i in range(2, 7)]
        + [f"BILL_AMT{i}" for i in range(1, 7)]
        + [f"PAY_AMT{i}" for i in range(1, 7)])

d = fetch_ucirepo(id=350)
df = d.data.features.copy()
assert df.shape[1] == 23, "Unexpected dataset shape"
df.columns = COLS
df["default"] = d.data.targets.iloc[:, 0].values
df.to_csv(OUT, index=False)
print(f"Saved {len(df):,} rows to {OUT}. Default rate: {df['default'].mean():.1%}")
