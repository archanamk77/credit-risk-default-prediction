"""Train, evaluate, explain and profit-optimise the default model."""
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from features import CATEGORICAL, TARGET, prepare

# ---- Business assumptions (NT$ per account). Change these and re-run. ----
PROFIT_GOOD = 1_000   # profit from approving a customer who repays
LOSS_BAD = 5_000      # loss from approving a customer who defaults
SEED = 42

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"

df = prepare(pd.read_csv(ROOT / "data" / "raw" / "credit_default.csv"))
X, y = df.drop(columns=[TARGET]), df[TARGET].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def evaluate(y_true, p):
    fpr, tpr, _ = roc_curve(y_true, p)
    prec, rec, _ = precision_recall_curve(y_true, p)
    ok = prec >= 0.5
    return {"roc_auc": roc_auc_score(y_true, p), "pr_auc": average_precision_score(y_true, p),
            "ks": float(np.max(tpr - fpr)), "recall_at_50pct_precision": float(rec[ok].max()) if ok.any() else 0.0}


def profit(y_true, p, t):
    approve = p < t
    return int((approve & (y_true == 0)).sum() * PROFIT_GOOD - (approve & (y_true == 1)).sum() * LOSS_BAD)


def credit_loss(y_true, p, t):
    return int(((p < t) & (y_true == 1)).sum() * LOSS_BAD)


# ---- 1. Baseline: logistic regression (class-weighted for the ~22% default imbalance) ----
pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
                        remainder=StandardScaler())
lr = Pipeline([("pre", pre), ("clf", LogisticRegression(class_weight="balanced", max_iter=2000))])
lr.fit(Xtr, ytr)
p_lr = lr.predict_proba(Xte)[:, 1]

# ---- 2. XGBoost with randomised hyper-parameter search (3-fold CV) ----
search = RandomizedSearchCV(
    XGBClassifier(eval_metric="logloss", tree_method="hist", n_jobs=-1, random_state=SEED),
    {"n_estimators": [200, 400], "max_depth": [3, 4, 5], "learning_rate": [0.03, 0.05, 0.1],
     "subsample": [0.7, 0.9], "colsample_bytree": [0.7, 0.9], "min_child_weight": [1, 5]},
    n_iter=12, cv=3, scoring="roc_auc", random_state=SEED, n_jobs=1)
search.fit(Xtr, ytr)
xgb = search.best_estimator_
p_xgb = xgb.predict_proba(Xte)[:, 1]
print("Best params:", search.best_params_)

metrics = {"logistic_regression": evaluate(yte, p_lr), "xgboost": evaluate(yte, p_xgb)}

# ---- 3. Profit-maximising threshold, chosen on out-of-fold TRAIN predictions (no test leakage) ----
oof = cross_val_predict(xgb, Xtr, ytr, cv=5, method="predict_proba")[:, 1]
grid = np.linspace(0.05, 0.95, 91)
t_best = float(grid[int(np.argmax([profit(ytr, oof, t) for t in grid]))])

approve_all = 1.01
loss_all, loss_model = credit_loss(yte, p_xgb, approve_all), credit_loss(yte, p_xgb, t_best)
metrics["business"] = {
    "threshold": t_best, "profit_per_good": PROFIT_GOOD, "loss_per_bad": LOSS_BAD,
    "test_profit_approve_all": profit(yte, p_xgb, approve_all),
    "test_profit_default_threshold_0.5": profit(yte, p_xgb, 0.5),
    "test_profit_optimised_threshold": profit(yte, p_xgb, t_best),
    "credit_loss_reduction_vs_approve_all_pct": round(100 * (1 - loss_model / loss_all), 1),
    "approval_rate": round(float((p_xgb < t_best).mean()), 3),
    "confusion_matrix_[[TN,FP],[FN,TP]]": confusion_matrix(yte, (p_xgb >= t_best).astype(int)).tolist(),
}
metrics = json.loads(json.dumps(metrics, default=float))
(ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))

# ---- 4. Plots ----
plt.figure(figsize=(6, 5))
for name, p in [("Logistic Regression", p_lr), ("XGBoost", p_xgb)]:
    f, t_, _ = roc_curve(yte, p); plt.plot(f, t_, label=f"{name} (AUC {roc_auc_score(yte, p):.3f})")
plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curve (test set)"); plt.legend(); plt.tight_layout(); plt.savefig(FIG / "roc.png", dpi=150); plt.close()

plt.figure(figsize=(6, 5))
for name, p in [("Logistic Regression", p_lr), ("XGBoost", p_xgb)]:
    pr, rc, _ = precision_recall_curve(yte, p); plt.plot(rc, pr, label=f"{name} (AP {average_precision_score(yte, p):.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-recall curve (test set)")
plt.legend(); plt.tight_layout(); plt.savefig(FIG / "pr_curve.png", dpi=150); plt.close()

plt.figure(figsize=(6, 5))
for name, p in [("Logistic Regression", p_lr), ("XGBoost", p_xgb)]:
    frac, mean = calibration_curve(yte, p, n_bins=10, strategy="quantile"); plt.plot(mean, frac, "o-", label=name)
plt.plot([0, 1], [0, 1], "k--", label="Perfect"); plt.xlabel("Predicted probability"); plt.ylabel("Actual default rate")
plt.title("Calibration curve"); plt.legend(); plt.tight_layout(); plt.savefig(FIG / "calibration.png", dpi=150); plt.close()

plt.figure(figsize=(7, 5))
plt.plot(grid, [profit(ytr, oof, t) for t in grid], label="Train (out-of-fold)")
plt.plot(grid, [profit(yte, p_xgb, t) for t in grid], label="Test")
plt.axvline(t_best, color="r", ls="--", label=f"Chosen threshold {t_best:.2f}")
plt.xlabel("Approval threshold (approve if default probability < threshold)"); plt.ylabel("Profit (NT$)")
plt.title("Profit vs threshold"); plt.legend(); plt.tight_layout(); plt.savefig(FIG / "profit_curve.png", dpi=150); plt.close()

tiers = pd.DataFrame({"p": p_xgb, "y": yte})
tiers["tier"] = pd.qcut(tiers["p"], [0, .5, .8, .95, 1], labels=["Low", "Medium", "High", "Very high"])
tier_tbl = tiers.groupby("tier", observed=True).agg(accounts=("y", "size"), avg_predicted_pd=("p", "mean"), actual_default_rate=("y", "mean"))
tier_tbl.round(3).to_csv(ROOT / "reports" / "risk_tiers.csv")
tier_tbl["actual_default_rate"].plot.bar(color=["green", "gold", "orange", "red"], figsize=(6, 5))
plt.ylabel("Actual default rate"); plt.title("Default rate by risk tier"); plt.xticks(rotation=0)
plt.tight_layout(); plt.savefig(FIG / "risk_tiers.png", dpi=150); plt.close()

sv = shap.TreeExplainer(xgb)(Xte.sample(min(2000, len(Xte)), random_state=SEED))
shap.plots.beeswarm(sv, max_display=12, show=False); plt.savefig(FIG / "shap_beeswarm.png", dpi=150, bbox_inches="tight"); plt.close()
shap.plots.bar(sv, max_display=12, show=False); plt.savefig(FIG / "shap_bar.png", dpi=150, bbox_inches="tight"); plt.close()

joblib.dump({"model": xgb, "threshold": t_best, "features": list(X.columns)}, ROOT / "models" / "model.joblib")
print("Done. Model saved.")
