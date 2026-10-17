import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from athena.tools.features import build_features, FEATURES, MARKET_FEATURES
from athena.universe import UNIVERSE

AS_OF = "2026-10-16"
DEV_END = "2026-01-01"   # 2026 is the hidden final exam
FINAL_TEST = False       # set to True ONLY ONCE, at the very end

PARAMS = dict(
    n_estimators=200, learning_rate=0.03, num_leaves=15,
    min_child_samples=200, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, random_state=42, verbose=-1,
)

RANK_FEATURES = [f + "_rank" for f in FEATURES]
MODEL_FEATURES = RANK_FEATURES + MARKET_FEATURES


def prepare_data():
    frames = []
    for t in UNIVERSE:
        df = build_features(t, AS_OF)
        if len(df) > 0:
            frames.append(df)
        else:
            print("Skipping (no prices):", t)
    data = pd.concat(frames).reset_index()
    data = data.dropna(subset=FEATURES + MARKET_FEATURES)
    print("Stocks used:", data["ticker"].nunique())

    # Rank each clue among the stocks on the same day (0 = lowest, 1 = highest)
    for f in FEATURES:
        data[f + "_rank"] = data.groupby("date")[f].rank(pct=True)

    # Clear winners (top 30%) = 1, clear losers (bottom 30%) = 0, middle skipped
    data["result_rank"] = data.groupby("date")["future_excess"].rank(pct=True)
    data["target"] = np.nan
    data.loc[data["result_rank"] >= 0.7, "target"] = 1
    data.loc[data["result_rank"] <= 0.3, "target"] = 0
    return data


def daily_scores(test):
    ic = test.groupby("date")[["p", "future_excess"]].apply(
        lambda g: g["p"].corr(g["future_excess"], method="spearman")
    )
    labelled = test.dropna(subset=["target"])
    auc = labelled.groupby("date")[["p", "target"]].apply(
        lambda g: roc_auc_score(g["target"], g["p"]) if g["target"].nunique() == 2 else np.nan
    )
    return ic.mean(), auc.mean()


data = prepare_data()
labelled = data.dropna(subset=["target", "future_excess"])
end_date = AS_OF if FINAL_TEST else DEV_END
print("Rows:", len(data), "| Clear winners/losers for training:", len(labelled))

results = []
for test_start in pd.date_range("2022-01-01", end_date, freq="3MS"):
    test_end = min(test_start + pd.DateOffset(months=3), pd.Timestamp(end_date))
    train_end = test_start - pd.Timedelta(days=30)

    train = labelled[labelled["date"] < train_end]
    test = data[(data["date"] >= test_start) & (data["date"] < test_end)]
    test = test.dropna(subset=["future_excess"]).copy()
    if len(test) == 0:
        continue

    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(train[MODEL_FEATURES], train["target"].astype(int))
    test["p"] = model.predict_proba(test[MODEL_FEATURES])[:, 1]

    ic, auc = daily_scores(test)
    print(f"{test_start.date()}: daily AUC = {auc:.3f}, rank IC = {ic:.3f}")
    results.append(test)

all_test = pd.concat(results)
ic, auc = daily_scores(all_test)
print(f"\nDEVELOPMENT (2022-2025): daily AUC = {auc:.3f}, rank IC = {ic:.3f}")

if FINAL_TEST:
    final_exam = all_test[all_test["date"] >= DEV_END]
    ic, auc = daily_scores(final_exam)
    print(f"FINAL EXAM (2026): daily AUC = {auc:.3f}, rank IC = {ic:.3f}")

all_test[["date", "ticker", "p", "future_excess", "target"]].to_csv(
    "data/forecast_walkforward_v2.csv", index=False
)

final = lgb.LGBMClassifier(**PARAMS)
final.fit(labelled[MODEL_FEATURES], labelled["target"].astype(int))
joblib.dump({"model": final, "features": MODEL_FEATURES}, "models/forecast_lgbm_v2.pkl")
print("Saved model to models/forecast_lgbm_v2.pkl")