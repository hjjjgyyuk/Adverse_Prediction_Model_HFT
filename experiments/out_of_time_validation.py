import os

import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier

from src.database import get_stock_data
from src.features import create_features
from src.labels import create_labels


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

os.makedirs("results", exist_ok=True)


def prepare_data(symbol):

    df = get_stock_data(symbol)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values("timestamp")

    df["timestamp"] = df["timestamp"].dt.floor("1s")

    df = (
        df.groupby("timestamp", as_index=False)
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

    df["ltp"] = df["ltp"].ffill(limit=5)
    df["volume"] = df["volume"].ffill(limit=5)

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

    df = df.dropna(
        subset=FEATURE_COLUMNS + ["adverse_buy"]
    )

    return df


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


def top_risk_analysis(
    y_test,
    probabilities
):

    result = pd.DataFrame({
        "actual": np.asarray(y_test),
        "probability": probabilities
    })

    result = result.sort_values(
        "probability",
        ascending=False
    )

    baseline = result["actual"].mean()

    rows = []

    for percentage in [
        0.1,
        0.5,
        1,
        2,
        5
    ]:

        n = max(
            1,
            int(
                len(result) *
                percentage /
                100
            )
        )

        top = result.head(n)

        rate = top["actual"].mean()

        lift = (
            rate / baseline
            if baseline > 0
            else np.nan
        )

        rows.append({
            "top_percent": percentage,
            "observations": n,
            "adverse_rate": rate,
            "baseline_rate": baseline,
            "lift": lift
        })

    return pd.DataFrame(rows)


def main():

    all_results = []
    all_precision = []

    for symbol in STOCKS:

        print("\n" + "=" * 70)
        print(f"STOCK: {symbol}")
        print("=" * 70)

        df = prepare_data(symbol)

        print(
            "Rows after preprocessing:",
            len(df)
        )

        print(
            "Start:",
            df["timestamp"].min()
        )

        print(
            "End:",
            df["timestamp"].max()
        )

        X = df[FEATURE_COLUMNS]
        y = df["adverse_buy"]

        n = len(df)

        split = int(
            n * 0.80
        )

        X_train = X.iloc[:split]
        y_train = y.iloc[:split]

        X_test = X.iloc[split:]
        y_test = y.iloc[split:]

        model = train_model(
            X_train,
            y_train
        )

        probabilities = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        auc = roc_auc_score(
            y_test,
            probabilities
        )

        pr_auc = average_precision_score(
            y_test,
            probabilities
        )

        baseline = y_test.mean()

        precision = top_risk_analysis(
            y_test,
            probabilities
        )

        precision.insert(
            0,
            "stock",
            symbol
        )

        all_precision.append(
            precision
        )

        print(
            f"ROC-AUC: {auc:.4f}"
        )

        print(
            f"PR-AUC: {pr_auc:.4f}"
        )

        print(
            f"Baseline adverse rate: "
            f"{baseline:.6%}"
        )

        print("\nTop-risk analysis:")
        print(precision)

        all_results.append({
            "stock": symbol,
            "rows": len(df),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "roc_auc": auc,
            "pr_auc": pr_auc,
            "baseline_adverse_rate": baseline
        })

    results = pd.DataFrame(
        all_results
    )

    precision = pd.concat(
        all_precision,
        ignore_index=True
    )

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        results[
            [
                "roc_auc",
                "pr_auc",
                "baseline_adverse_rate"
            ]
        ].mean()
    )

    print("\nMean ROC-AUC:")
    print(
        results["roc_auc"].mean()
    )

    print("\nMedian ROC-AUC:")
    print(
        results["roc_auc"].median()
    )

    print("\nMean PR-AUC:")
    print(
        results["pr_auc"].mean()
    )

    print("\nMean top-risk lift:")

    print(
        precision
        .groupby("top_percent")["lift"]
        .mean()
    )

    results.to_csv(
        "results/final_oot_results.csv",
        index=False
    )

    precision.to_csv(
        "results/final_risk_concentration.csv",
        index=False
    )

    print("\nSaved:")
    print(
        "results/final_oot_results.csv"
    )

    print(
        "results/final_risk_concentration.csv"
    )


if __name__ == "__main__":
    main()