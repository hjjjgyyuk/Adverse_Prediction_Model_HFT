

import pandas as pd
import numpy as np

from src.database import get_stock_data
from src.features import create_features
from src.labels import create_labels

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)


STOCKS = [
    "AXISBANK",
    "RELIANCE",
    "INFY",
    "TCS",
    "ICICIBANK",
    "HDFCBANK",
    "SBIN",
    "ITC",
    "LT",
    "MARUTI"
]

HORIZON = 5
THRESHOLD = 0.001


FEATURE_COLUMNS = [
    "return_1s",
    "return_2s",
    "return_5s",
    "return_10s",
    "return_20s",
    "return_30s",
    "return_60s",
    "momentum_acceleration",
    "volatility_10s",
    "volatility_30s",
    "volatility_60s",
    "volume_change",
    "volume_10s",
    "volume_30s",
    "volume_change_pct",
    "up_moves_10s",
    "down_moves_10s"
]


def train_model(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def evaluate(model, X_test, y_test):

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    cutoff = max(
        1,
        int(len(probabilities) * 0.005)
    )

    top_indices = np.argsort(
        probabilities
    )[-cutoff:]

    top_precision = (
        y_test.iloc[top_indices].mean()
    )

    baseline_rate = y_test.mean()

    lift = (
        top_precision / baseline_rate
        if baseline_rate > 0
        else np.nan
    )

    return auc, pr_auc, top_precision, lift


results = []


for stock in STOCKS:

    print(f"\n{'=' * 60}")
    print(f"STOCK: {stock}")
    print(f"{'=' * 60}")

    df = get_stock_data(stock)

    if len(df) == 0:
        continue

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = (
        df.set_index("timestamp")
        .resample("1s")
        .last()
        .ffill()
        .reset_index()
    )

    df = create_features(df)

    df = create_labels(
        df,
        horizon_seconds=HORIZON,
        threshold=THRESHOLD
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna()

    split = int(len(df) * 0.8)

    train = df.iloc[:split]
    test = df.iloc[split:]

    X_train = train[FEATURE_COLUMNS]
    X_test = test[FEATURE_COLUMNS]

    y_train = train["adverse_buy"]
    y_test = test["adverse_buy"]

    model = train_model(
        X_train,
        y_train
    )

    auc, pr_auc, precision, lift = evaluate(
        model,
        X_test,
        y_test
    )

    print(f"ROC-AUC: {auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(
        f"Top 0.5% precision: "
        f"{precision:.4f}"
    )
    print(
        f"Top 0.5% lift: "
        f"{lift:.2f}x"
    )

    results.append({
        "stock": stock,
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "top_0.5_precision": precision,
        "top_0.5_lift": lift
    })


results_df = pd.DataFrame(results)

results_df.to_csv(
    "experiments/random_forest_results.csv",
    index=False
)


print("\nFINAL RANDOM FOREST RESULTS")
print("=" * 60)

print(
    results_df[
        [
            "roc_auc",
            "pr_auc",
            "top_0.5_precision",
            "top_0.5_lift"
        ]
    ].mean()
)

print("\nPer-stock results:")
print(results_df)

print(
    "\nSaved to "
    "experiments/random_forest_results.csv"
)