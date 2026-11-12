"""Exploratory analysis: saves figures to reports/figures."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from features import prepare, TARGET

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
df = prepare(pd.read_csv(ROOT / "data" / "raw" / "credit_default.csv"))
print("Rows:", len(df), "| Default rate:", round(df[TARGET].mean(), 3))
print("Missing values:", int(df.isna().sum().sum()))

df["age_band"] = pd.cut(df["AGE"], [20, 30, 40, 50, 60, 80])
df["limit_band"] = pd.qcut(df["LIMIT_BAL"], 5, duplicates="drop")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.ravel(), ["EDUCATION", "MARRIAGE", "age_band", "PAY_1"]):
    df.groupby(col, observed=True)[TARGET].mean().plot.bar(ax=ax, color="steelblue")
    ax.set_title(f"Default rate by {col}"); ax.set_ylabel("Default rate")
plt.tight_layout(); plt.savefig(FIG / "eda_default_rates.png", dpi=150); plt.close()

pivot = df.pivot_table(TARGET, index="PAY_1", columns="limit_band", aggfunc="mean", observed=True)
plt.figure(figsize=(9, 6)); sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Reds")
plt.title("Default rate: latest repayment status vs credit limit band")
plt.tight_layout(); plt.savefig(FIG / "eda_heatmap.png", dpi=150); plt.close()

num = df.select_dtypes("number")
plt.figure(figsize=(6, 10)); sns.heatmap(num.corr()[[TARGET]].sort_values(TARGET, ascending=False), annot=True, cmap="coolwarm")
plt.title("Correlation with default"); plt.tight_layout(); plt.savefig(FIG / "eda_correlation.png", dpi=150); plt.close()
print("Saved EDA figures.")
