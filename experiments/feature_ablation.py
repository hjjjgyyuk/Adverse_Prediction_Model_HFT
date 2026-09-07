import pandas as pd
import numpy as np

from src.database import get_stock_data
from src.features import create_features
from src.labels import create_labels

from xgboost import XGBClassifier

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


FEATURE_GROUPS = {

    "returns": [
        "return_1s",
        "return_2s",
        "return_5s",
        "return_10s",
        "return_20s",
        "return_30s",
        "return_60s"
    ],

    "volatility": [
        "volatility_10s",
        "volatility_30s",
        "volatility_60s"
    ],

    "volume": [
        "volume_change",
        "volume_10s",
        "volume_30s",
        "volume_change_pct"
    ],

    "returns_volatility": [
        "return_1s",
        "return_2s",
        "return_5s",
        "return_10s",
        "return_20s",
        "return_30s",
        "return_60s",
        "volatility_10s",
        "volatility_30s",
        "volatility_60s"
    ],

    "returns_volume": [
        "return_1s",
        "return_2s",
        "return_5s",
        "return_10s",
        "return_20s",
        "return_30s",
        "return_60s",
        "volume_change",
        "volume_10s",
        "volume_30s",
        "volume_change_pct"
    ],

    "all_features": [
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
}


def train_model(X_train, y_train):

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def evaluate(
    model,
    X_test,
    y_test
):

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
        int(
            len(probabilities)
            * 0.005
        )
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

    return (
        auc,
        pr_auc,
        top_precision,
        lift
    )


results = []


for stock in STOCKS:

    print(
        f"\n{'=' * 60}"
    )

    print(
        f"STOCK: {stock}"
    )

    print(
        f"{'=' * 60}"
    )

    df = get_stock_data(stock)

    if len(df) == 0:
        print("No data found.")
        continue

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    )

    df["timestamp"] = (
        df["timestamp"].dt.floor("1s")
    )

    df = (
        df.groupby(
            "timestamp",
            as_index=False
        )
        .agg(
            ltp=("ltp", "last"),
            volume=("volume", "last")
        )
    )

    df = (
        df.set_index("timestamp")
        .resample("1s")
        .last()
    )

    df["ltp"] = df["ltp"].ffill(
        limit=5
    )

    df["volume"] = df["volume"].ffill(
        limit=5
    )

    df = df.reset_index()

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

    all_features = list({
        feature
        for features in FEATURE_GROUPS.values()
        for feature in features
    })

    df = df.dropna(
        subset=all_features + [
            "adverse_buy"
        ]
    )

    split = int(
        len(df) * 0.8
    )

    train = df.iloc[:split]
    test = df.iloc[split:]

    print(
        f"Train: {len(train)} | "
        f"Test: {len(test)}"
    )

    y_train = train[
        "adverse_buy"
    ]

    y_test = test[
        "adverse_buy"
    ]

    for group_name, features in FEATURE_GROUPS.items():

        X_train = train[features]
        X_test = test[features]

        model = train_model(
            X_train,
            y_train
        )

        (
            auc,
            pr_auc,
            precision,
            lift
        ) = evaluate(
            model,
            X_test,
            y_test
        )

        print(
            f"{group_name:<22}"
            f"AUC: {auc:.4f} | "
            f"PR-AUC: {pr_auc:.4f} | "
            f"Top 0.5% Lift: {lift:.2f}x"
        )

        results.append({
            "stock": stock,
            "feature_set": group_name,
            "roc_auc": auc,
            "pr_auc": pr_auc,
            "top_0.5_precision": precision,
            "top_0.5_lift": lift
        })


results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    "experiments/feature_ablation_results.csv",
    index=False
)


print("\n\nFINAL SUMMARY")
print("=" * 70)

summary = (
    results_df
    .groupby("feature_set")
    .agg(
        mean_auc=("roc_auc", "mean"),
        median_auc=("roc_auc", "median"),
        mean_pr_auc=("pr_auc", "mean"),
        mean_top_0_5_precision=(
            "top_0.5_precision",
            "mean"
        ),
        mean_top_0_5_lift=(
            "top_0.5_lift",
            "mean"
        )
    )
    .sort_values(
        "mean_auc",
        ascending=False
    )
)

print(summary)

print(
    "\nSaved to "
    "experiments/feature_ablation_results.csv"
)