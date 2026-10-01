"""
model_comparison.py - PRT661 Life Expectancy Prediction (Sydney Group 3, Theme 2)
Assessment 3: delivers the next steps planned in Assessment 2.

  1. VIF analysis for multicollinearity (drop features with VIF > 10 for the linear model)
  2. Log-transformed GDP and Population (created in data_pipeline.py)
  3. Random Forest and XGBoost, tuned with randomised search
  4. Two evaluations:
       a) random 80:20 split, random_state=42  -> directly comparable with the A2 baseline
       b) 5-fold GroupKFold by country         -> no country appears in both train and test
  Median imputation sits inside every pipeline, so it is fitted on training folds only.

Usage:  python model_comparison.py      (after data_pipeline.py)
Output: outputs/model_metrics.csv, outputs/vif.csv, outputs/*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

SEED = 42
DATA = Path("data/processed_data.csv")
OUT = Path("outputs")
TARGET = "Life expectancy"
DROP = [TARGET, "Country", "Status", "GDP", "Population"]   # raw GDP/Pop replaced by log versions
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
X = df.drop(columns=[c for c in DROP if c in df.columns])
y = df[TARGET]
groups = df["Country"]


# ---------- 1. VIF analysis ----------
def vif_table(frame):
    """VIF_i = 1 / (1 - R2_i), regressing each feature on all the others."""
    filled = frame.fillna(frame.median())
    out = {}
    for col in filled.columns:
        others = filled.drop(columns=col)
        r2 = LinearRegression().fit(others, filled[col]).score(others, filled[col])
        out[col] = np.inf if r2 >= 1 else 1 / (1 - r2)
    return pd.Series(out).sort_values(ascending=False)


vif_all = vif_table(X)
lin_features = list(X.columns)
while True:                                   # drop the worst feature until all VIF <= 10
    v = vif_table(X[lin_features])
    if v.iloc[0] <= 10:
        break
    lin_features.remove(v.index[0])
dropped = [c for c in X.columns if c not in lin_features]
vif_all.round(2).to_csv(OUT / "vif.csv", header=["VIF"])
print("VIF > 10 removed for the linear model:", dropped)


# ---------- 2. Models ----------
class Cols(BaseEstimator, TransformerMixin):
    """Select a subset of columns inside a pipeline."""
    def __init__(self, cols=None): self.cols = cols
    def fit(self, X, y=None): return self
    def transform(self, X): return X[self.cols]


def median(): return SimpleImputer(strategy="median")


models = {
    "Linear Regression (A2 baseline)": make_pipeline(median(), LinearRegression()),
    "Linear Regression (VIF-reduced)": make_pipeline(Cols(lin_features), median(), LinearRegression()),
    "Random Forest (tuned)": RandomizedSearchCV(
        make_pipeline(median(), RandomForestRegressor(random_state=SEED, n_jobs=-1)),
        {"randomforestregressor__n_estimators": [200, 400],
         "randomforestregressor__max_depth": [None, 10, 20, 30],
         "randomforestregressor__min_samples_leaf": [1, 2, 4],
         "randomforestregressor__max_features": [0.33, 0.5, 1.0]},
        n_iter=8, cv=GroupKFold(n_splits=5), scoring="neg_root_mean_squared_error",
        random_state=SEED, n_jobs=-1),
    "XGBoost (tuned)": RandomizedSearchCV(
        make_pipeline(median(), XGBRegressor(random_state=SEED, n_jobs=-1)),
        {"xgbregressor__n_estimators": [300, 600],
         "xgbregressor__learning_rate": [0.03, 0.05, 0.1],
         "xgbregressor__max_depth": [3, 4, 6, 8],
         "xgbregressor__subsample": [0.7, 0.85, 1.0],
         "xgbregressor__colsample_bytree": [0.6, 0.8, 1.0]},
        n_iter=8, cv=GroupKFold(n_splits=5), scoring="neg_root_mean_squared_error",
        random_state=SEED, n_jobs=-1),
}


def score(y_true, pred):
    return (np.sqrt(mean_squared_error(y_true, pred)), mean_absolute_error(y_true, pred), r2_score(y_true, pred))


# ---------- 3a. Random 80:20 split (same as A2) ----------
X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(X, y, groups, test_size=0.2, random_state=SEED)
rows, preds, fitted = [], {}, {}
for name, m in models.items():
    fit_kw = {"groups": g_tr} if isinstance(m, RandomizedSearchCV) else {}
    m.fit(X_tr, y_tr, **fit_kw)
    fitted[name] = m.best_estimator_ if isinstance(m, RandomizedSearchCV) else m
    preds[name] = fitted[name].predict(X_te)
    rmse, mae, r2 = score(y_te, preds[name])
    rows.append({"Model": name, "Split_RMSE": rmse, "Split_MAE": mae, "Split_R2": r2})
    if isinstance(m, RandomizedSearchCV):
        print(f"{name} best params: {m.best_params_}")

# ---------- 3b. 5-fold GroupKFold by country (honest, unseen countries) ----------
gkf = GroupKFold(n_splits=5)
for row, name in zip(rows, models):
    cv = cross_validate(fitted[name], X, y, groups=groups, cv=gkf,
                        scoring=("neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"))
    row["GroupCV_RMSE"] = -cv["test_neg_root_mean_squared_error"].mean()
    row["GroupCV_RMSE_sd"] = cv["test_neg_root_mean_squared_error"].std()
    row["GroupCV_MAE"] = -cv["test_neg_mean_absolute_error"].mean()
    row["GroupCV_R2"] = cv["test_r2"].mean()

metrics = pd.DataFrame(rows).round(3)
metrics.to_csv(OUT / "model_metrics.csv", index=False)
pd.set_option("display.width", 200)
print("\n", metrics.to_string(index=False))
best = metrics.loc[metrics["GroupCV_RMSE"].idxmin(), "Model"]
print(f"\nBest model (lowest grouped-CV RMSE): {best}")

# ---------- 4. Charts ----------
short = [n.split(" (")[0].replace("Linear Regression", "LR") + ("\n(VIF)" if "VIF" in n else "") for n in metrics["Model"]]
colors = ["#9aa5b1", "#c3cbd3", "#2a7ab0", "#1f9d6b"]

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
xi = np.arange(len(metrics)); w = 0.38
ax[0].bar(xi - w / 2, metrics["Split_RMSE"], w, label="Random 80:20 split (A2 method)", color="#9fc5e8")
ax[0].bar(xi + w / 2, metrics["GroupCV_RMSE"], w, yerr=metrics["GroupCV_RMSE_sd"], capsize=3,
          label="5-fold by country (unseen countries)", color="#1f4e79")
ax[0].set_xticks(xi, short); ax[0].set_ylabel("RMSE (years, lower is better)")
ax[0].set_title("Prediction error by model"); ax[0].legend(frameon=False, fontsize=8)
ax[1].bar(xi, metrics["GroupCV_R2"], color=colors)
for i, v in enumerate(metrics["GroupCV_R2"]):
    ax[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
ax[1].set_xticks(xi, short); ax[1].set_ylim(0, 1.05)
ax[1].set_title("R² on unseen countries (5-fold by country)")
for a in ax: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / "model_comparison.png", dpi=150); plt.close()

tree = best if "Linear" not in best else "XGBoost (tuned)"
est = fitted[tree][-1]
imp = pd.Series(est.feature_importances_, index=X.columns).nlargest(10)[::-1]
plt.figure(figsize=(7, 4.5))
plt.barh(imp.index, imp.values, color="#2a7ab0")
plt.title(f"Top 10 drivers of life expectancy - {tree}")
plt.xlabel("Feature importance"); plt.gca().spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / "feature_importance.png", dpi=150); plt.close()

plt.figure(figsize=(5.5, 5))
plt.scatter(y_te, preds[best], alpha=0.5, s=14, color="#2a7ab0")
lims = [y_te.min() - 2, y_te.max() + 2]
plt.plot(lims, lims, "--", color="grey", linewidth=1)
plt.xlabel("Actual life expectancy (years)"); plt.ylabel("Predicted (years)")
plt.title(f"Predicted vs actual - {best}"); plt.gca().spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / "predicted_vs_actual.png", dpi=150); plt.close()
print(f"Charts saved to {OUT}/")
