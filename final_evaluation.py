"""
final_evaluation.py - PRT661 Life Expectancy Prediction (Sydney Group 3, Theme 2)
Assessment 4: corrected validation, following the Assessment 3 feedback.

1. Nested country-based validation: VIF feature selection and hyperparameter
   tuning both happen INSIDE each outer training fold (no information from the
   test countries is used to choose features or settings).
2. Forecast test: train on 2000-2012, test on the later years 2013-2015.
3. Model without mortality indicators (adult mortality, infant deaths,
   under-five deaths, HIV/AIDS deaths), which are closely tied to the outcome
   and may not be available for a genuine future prediction.
4. Error by development status and by region, and 90% prediction intervals.
5. Trains and saves the final model for inference (predict.py, dashboard.py).

Usage:  python final_evaluation.py      (needs data/processed_data.csv)
"""
import json
from pathlib import Path

import country_converter as coco
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

SEED = 42
DATA = Path("data/processed_data.csv")
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
TARGET = "Life expectancy"
DROP = [TARGET, "Country", "Status", "GDP", "Population"]
MORTALITY = ["Adult Mortality", "infant deaths", "under-five deaths", "HIV/AIDS"]

df = pd.read_csv(DATA)
df["Region"] = coco.convert(df["Country"].tolist(), to="continent", not_found="Other")
X = df.drop(columns=[c for c in DROP if c in df.columns] + ["Region"])
y = df[TARGET]
groups = df["Country"]


# ---------- VIF selection, fitted on training data only ----------
class VIFSelector(BaseEstimator, TransformerMixin):
    """Drops the feature with the highest VIF until all VIF <= threshold.
    Fitted inside the pipeline, so it only ever sees the training fold."""
    def __init__(self, threshold=10.0): self.threshold = threshold

    def fit(self, X, y=None):
        A = np.asarray(X, dtype=float); keep = list(range(A.shape[1]))
        while len(keep) > 1:
            vifs = []
            for j in keep:
                others = [k for k in keep if k != j]
                r2 = LinearRegression().fit(A[:, others], A[:, j]).score(A[:, others], A[:, j])
                vifs.append(np.inf if r2 >= 1 else 1 / (1 - r2))
            worst = int(np.argmax(vifs))
            if vifs[worst] <= self.threshold: break
            keep.pop(worst)
        self.keep_ = keep
        return self

    def transform(self, X): return np.asarray(X, dtype=float)[:, self.keep_]


def median(): return SimpleImputer(strategy="median")


def tuned(estimator, grid):
    """Hyperparameter search with an inner GroupKFold by country."""
    return RandomizedSearchCV(estimator, grid, n_iter=8, cv=GroupKFold(n_splits=5),
                              scoring="neg_root_mean_squared_error", random_state=SEED, n_jobs=-1)


RF_GRID = {"randomforestregressor__n_estimators": [200, 400],
           "randomforestregressor__max_depth": [None, 10, 20, 30],
           "randomforestregressor__min_samples_leaf": [1, 2, 4],
           "randomforestregressor__max_features": [0.33, 0.5, 1.0]}
XGB_GRID = {"xgbregressor__n_estimators": [300, 600],
            "xgbregressor__learning_rate": [0.03, 0.05, 0.1],
            "xgbregressor__max_depth": [3, 4, 6, 8],
            "xgbregressor__subsample": [0.7, 0.85, 1.0],
            "xgbregressor__colsample_bytree": [0.6, 0.8, 1.0]}

MODELS = {
    "Linear Regression": lambda: make_pipeline(median(), LinearRegression()),
    "Linear Regression (VIF in fold)": lambda: make_pipeline(median(), VIFSelector(10), LinearRegression()),
    "Random Forest (tuned in fold)": lambda: tuned(make_pipeline(median(), RandomForestRegressor(random_state=SEED, n_jobs=-1)), RF_GRID),
    "XGBoost (tuned in fold)": lambda: tuned(make_pipeline(median(), XGBRegressor(random_state=SEED, n_jobs=-1)), XGB_GRID),
}


def fit(model, Xtr, ytr, gtr):
    if isinstance(model, RandomizedSearchCV): model.fit(Xtr, ytr, groups=gtr)
    else: model.fit(Xtr, ytr)
    return model


def metrics(y_true, pred):
    return {"RMSE": np.sqrt(mean_squared_error(y_true, pred)),
            "MAE": mean_absolute_error(y_true, pred), "R2": r2_score(y_true, pred)}


# ---------- 1. Nested country-based validation ----------
outer = GroupKFold(n_splits=5)
nested_rows, oof = [], {}


def nested(name, make, Xd):
    pred = pd.Series(np.nan, index=y.index); folds = []
    for tr, te in outer.split(Xd, y, groups):
        m = fit(make(), Xd.iloc[tr], y.iloc[tr], groups.iloc[tr])
        pred.iloc[te] = m.predict(Xd.iloc[te])
        folds.append(metrics(y.iloc[te], pred.iloc[te]))
    f = pd.DataFrame(folds)
    nested_rows.append({"Model": name, "RMSE": f.RMSE.mean(), "RMSE_sd": f.RMSE.std(ddof=0),
                        "MAE": f.MAE.mean(), "R2": f.R2.mean()})
    oof[name] = pred
    print(f"nested  {name:42s} RMSE {f.RMSE.mean():.2f} ± {f.RMSE.std(ddof=0):.2f}  R2 {f.R2.mean():.3f}")


for name, make in MODELS.items():
    nested(name, make, X)
X_nomort = X.drop(columns=MORTALITY)
nested("XGBoost without mortality indicators", MODELS["XGBoost (tuned in fold)"], X_nomort)
pd.DataFrame(nested_rows).round(3).to_csv(OUT / "nested_metrics.csv", index=False)

# ---------- 2. Forecast test: train 2000-2012, test 2013-2015 ----------
train_t, test_t = df["Year"] <= 2012, df["Year"] >= 2013
fc_rows = []
for name, make, Xd in [*[(n, m, X) for n, m in MODELS.items()],
                       ("XGBoost without mortality indicators", MODELS["XGBoost (tuned in fold)"], X_nomort)]:
    m = fit(make(), Xd[train_t], y[train_t], groups[train_t])
    fc_rows.append({"Model": name, **metrics(y[test_t], m.predict(Xd[test_t]))})
    print(f"forecast {name:42s} RMSE {fc_rows[-1]['RMSE']:.2f}  R2 {fc_rows[-1]['R2']:.3f}")
pd.DataFrame(fc_rows).round(3).to_csv(OUT / "forecast_metrics.csv", index=False)

# ---------- 3. Error by group and 90% prediction intervals (final XGBoost) ----------
best = "XGBoost (tuned in fold)"
res = df[["Country", "Year", "Status", "Region", TARGET]].rename(columns={TARGET: "Actual"})
res["Predicted"] = oof[best]
res["AbsError"] = (res.Predicted - res.Actual).abs()
q90 = res.groupby("Status")["AbsError"].quantile(0.9).to_dict()        # interval half-width per status
res["Lower"] = res.Predicted - res.Status.map(q90)
res["Upper"] = res.Predicted + res.Status.map(q90)
res["Covered"] = res.Actual.between(res.Lower, res.Upper)
grp = []
for col in ["Status", "Region"]:
    for k, g in res.groupby(col):
        grp.append({"Group type": col, "Group": k, "Countries": g.Country.nunique(),
                    "MAE": g.AbsError.mean(), "RMSE": np.sqrt((g.AbsError ** 2).mean())})
pd.DataFrame(grp).round(2).to_csv(OUT / "error_by_group.csv", index=False)
res.round(2).to_csv(OUT / "predictions.csv", index=False)
print("\n90% interval half-width by status:", {k: round(v, 2) for k, v in q90.items()},
      "| coverage:", round(res.Covered.mean(), 3))
print(pd.DataFrame(grp).round(2).to_string(index=False))

# ---------- 4. Final model on all data, saved for inference ----------
final = fit(MODELS[best](), X, y, groups)
joblib.dump(final.best_estimator_, OUT / "final_model.joblib")
json.dump({"features": list(X.columns), "interval_90": {k: round(v, 3) for k, v in q90.items()},
           "best_params": final.best_params_}, open(OUT / "final_model_info.json", "w"), indent=2)
print("\nSaved outputs/final_model.joblib and final_model_info.json")

# ---------- 5. Charts ----------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
imp = pd.Series(final.best_estimator_[-1].feature_importances_, index=X.columns).nlargest(10)[::-1]
plt.figure(figsize=(7, 4.5)); plt.barh(imp.index, imp.values, color="#2a7ab0")
plt.title("Top 10 drivers of life expectancy - final XGBoost"); plt.xlabel("Feature importance")
plt.gca().spines[["top", "right"]].set_visible(False); plt.tight_layout()
plt.savefig(OUT / "feature_importance.png", dpi=150); plt.close()

n = pd.DataFrame(nested_rows).set_index("Model")["RMSE"]; f = pd.DataFrame(fc_rows).set_index("Model")["RMSE"]
labels = ["LR", "LR\n(VIF)", "Random\nForest", "XGBoost", "XGBoost\nno mortality"]
x = np.arange(len(n)); plt.figure(figsize=(8, 4.2))
plt.bar(x - 0.2, n.values, 0.4, color="#1f4e79", label="Unseen countries (nested 5-fold)")
plt.bar(x + 0.2, f.values, 0.4, color="#9dc3e6", label="Later years (train 2000-12, test 2013-15)")
for i, (a, b) in enumerate(zip(n.values, f.values)):
    plt.text(i - 0.2, a + 0.05, f"{a:.2f}", ha="center", fontsize=8); plt.text(i + 0.2, b + 0.05, f"{b:.2f}", ha="center", fontsize=8)
plt.xticks(x, labels); plt.ylabel("RMSE (years, lower is better)"); plt.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
plt.title("Prediction error: unseen countries vs later years"); plt.gca().spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig(OUT / "final_model_comparison.png", dpi=150); plt.close()
print("Saved charts")
