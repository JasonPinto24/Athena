import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

with open("config.yaml") as f:
    W = yaml.safe_load(f)["weights"]

# ---------- 1. Load history ----------
hist = pd.read_csv("data/history.csv", parse_dates=["date"])
hist = hist.dropna(subset=["future_excess"])

SIGNALS = ["F", "M", "vol", "mdd", "beta"]
if hist["S"].notna().sum() > 50:
    SIGNALS.append("S")
print("Signals used:", SIGNALS)

# ---------- 2. Same scale: z-score each signal within each month ----------
data = hist.copy()
for c in SIGNALS:
    g = data.groupby("date")[c]
    z = (data[c] - g.transform("mean")) / g.transform("std")
    data[c + "_z"] = z.clip(-3, 3).fillna(0.0)          # missing = average (neutral)
data["F_missing"] = hist["F"].isna().astype(int)
FEATURES = [c + "_z" for c in SIGNALS] + ["F_missing"]
if "S" in SIGNALS:
    data["S_missing"] = hist["S"].isna().astype(int)
    FEATURES.append("S_missing")


def make_lr():
    return LogisticRegression(C=0.5, max_iter=1000)


def make_gb():
    return lgb.LGBMClassifier(n_estimators=150, learning_rate=0.03, num_leaves=7,
                              min_child_samples=40, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, random_state=42, verbose=-1)


# ---------- 3. Walk-forward: train on the past, skip a month, predict the next ----------
months = sorted(data["date"].unique())
results = []
for i, m in enumerate(months):
    if i < 12:
        continue
    train = data[(data["date"] < months[i - 1]) & data["target"].notna()]
    test = data[data["date"] == m].copy()

    lr = make_lr().fit(train[FEATURES], train["target"].astype(int))
    gb = make_gb().fit(train[FEATURES], train["target"].astype(int))

    test["p_lr"] = lr.predict_proba(test[FEATURES])[:, 1]
    test["p_gb"] = gb.predict_proba(test[FEATURES])[:, 1]
    s_part = W["sentiment"] * test["S_z"] if "S" in SIGNALS else 0.0
    test["p_formula"] = W["fundamentals"] * test["F_z"] + W["forecast"] * test["M_z"] + s_part
    test["p_forecast"] = test["M_z"]
    results.append(test)

oos = pd.concat(results)
print(f"Test months: {oos['date'].nunique()} | Test rows: {len(oos)}")


# ---------- 4. Score every approach ----------
def evaluate(df, col):
    ic = df.groupby("date")[[col, "future_excess"]].apply(
        lambda g: g[col].corr(g["future_excess"], method="spearman"))
    lab = df.dropna(subset=["target"])
    auc = lab.groupby("date")[[col, "target"]].apply(
        lambda g: roc_auc_score(g["target"], g[col]) if g["target"].nunique() == 2 else np.nan)

    def spread(g):
        q = g[col].rank(pct=True)
        return g.loc[q >= 0.8, "future_excess"].mean() - g.loc[q <= 0.2, "future_excess"].mean()

    spr = df.groupby("date")[[col, "future_excess"]].apply(spread)
    return {"IC": ic.mean(), "IC positive %": (ic > 0).mean(), "AUC": auc.mean(), "Spread": spr.mean()}


table = pd.DataFrame({
    "Forecast only": evaluate(oos, "p_forecast"),
    "Old formula": evaluate(oos, "p_formula"),
    "A: Logistic": evaluate(oos, "p_lr"),
    "B: LightGBM": evaluate(oos, "p_gb"),
}).T
print("\n========== WALK-FORWARD RESULTS (2024-2025) ==========")
print(table.round(3).to_string())

# ---------- 5. Calibration ladder: do higher probabilities really do better? ----------
oos["bucket"] = pd.qcut(oos["p_lr"], 5, labels=["1 (lowest)", "2", "3", "4", "5 (highest)"])
ladder = oos.groupby("bucket", observed=True).agg(
    avg_prob=("p_lr", "mean"),
    avg_excess_return=("future_excess", "mean"),
    share_top30=("target", lambda t: (t == 1).sum() / len(t)),
)
print("\n========== CALIBRATION LADDER (Model A) ==========")
print(ladder.round(3).to_string())

# ---------- 6. Final model on ALL history + decision cut-offs ----------
train_all = data[data["target"].notna()]
final = make_lr().fit(train_all[FEATURES], train_all["target"].astype(int))
p_all = final.predict_proba(data[FEATURES])[:, 1]
buy_p, sell_p = float(np.quantile(p_all, 0.75)), float(np.quantile(p_all, 0.25))

weights = pd.Series(final.coef_[0], index=FEATURES).sort_values(ascending=False)
print("\n========== LEARNED WEIGHTS (Model A) ==========")
print(weights.round(3).to_string())
print(f"\nDecision cut-offs: Buy if p >= {buy_p:.3f} | Sell if p <= {sell_p:.3f}")

joblib.dump({"model": final, "features": FEATURES, "signals": SIGNALS,
             "buy_p": buy_p, "sell_p": sell_p}, "models/meta_model.pkl")
oos.drop(columns=["bucket"]).to_csv("data/meta_walkforward.csv", index=False)
print("Saved models/meta_model.pkl and data/meta_walkforward.csv")