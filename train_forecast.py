import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score

from athena.tools.features import build_features, FEATURES

TICKERS = [
    "INFY.NS", "TCS.NS", "WIPRO.NS", "HCLTECH.NS",
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS",
    "SBIN.NS", "AXISBANK.NS", "ITC.NS", "HINDUNILVR.NS",
    "NESTLEIND.NS", "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS",
    "MARUTI.NS", "M&M.NS", "TATASTEEL.NS", "LT.NS",
]
AS_OF = "2026-10-16"

PARAMS = dict(
    n_estimators=200,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=200,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

# 1. Build one big table: all stocks, all days
frames = [build_features(t, AS_OF) for t in TICKERS]
data = pd.concat(frames)
data = data.dropna(subset=FEATURES + ["label"]).sort_index()
data["label"] = data["label"].astype(int)
print("Training rows:", len(data))

# 2. Walk-forward: train on the past, test on the next 3 months
results = []
for test_start in pd.date_range("2022-01-01", data.index.max(), freq="3MS"):
    test_end = test_start + pd.DateOffset(months=3)
    train_end = test_start - pd.Timedelta(days=30)

    train = data[data.index < train_end]
    test = data[(data.index >= test_start) & (data.index < test_end)]
    if len(test) == 0:
        continue

    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(train[FEATURES], train["label"])
    p = model.predict_proba(test[FEATURES])[:, 1]

    auc = roc_auc_score(test["label"], p)
    print(f"Test {test_start.date()} to {test_end.date()}: AUC = {auc:.3f}")
    results.append(pd.DataFrame(
        {"p": p, "label": test["label"].values, "ticker": test["ticker"].values},
        index=test.index,
    ))

preds = pd.concat(results)
print("Overall walk-forward AUC:", round(roc_auc_score(preds["label"], preds["p"]), 3))
preds.to_csv("data/forecast_walkforward.csv")

# 3. Final model on ALL data (used by the dashboard for today's prediction)
final = lgb.LGBMClassifier(**PARAMS)
final.fit(data[FEATURES], data["label"])
joblib.dump(final, "models/forecast_lgbm.pkl")
print("Saved model to models/forecast_lgbm.pkl")

# 4. Which clues mattered most?
importance = pd.Series(final.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(importance)