import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss
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
os.makedirs("plots", exist_ok=True)


def prepare_data(symbol):

    df = get_stock_data(symbol)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = df.sort_values("timestamp")

    df["timestamp"] = (
        df["timestamp"]
        .dt.floor("1s")
    )

    df = (
        df.groupby("timestamp")
        .agg(
            ltp=("ltp", "last"),
            volume=("volume", "last")
        )
        .reset_index()
    )

    df = create_features(df)

    df = create_labels(
        df,
        horizon_seconds=HORIZON,
        threshold=THRESHOLD
    )

    df = df.dropna(
        subset=FEATURE_COLUMNS + ["adverse_buy"]
    )

    return df


def train_xgboost(X_train, y_train):

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def calibrate_probabilities(raw_probabilities, y_calibration):

    raw_probabilities = np.clip(
        raw_probabilities,
        1e-6,
        1 - 1e-6
    )

    logits = np.log(
        raw_probabilities /
        (1 - raw_probabilities)
    )

    calibrator = LogisticRegression(
        max_iter=1000
    )

    calibrator.fit(
        logits.reshape(-1, 1),
        y_calibration
    )

    return calibrator


def apply_calibration(calibrator, probabilities):

    probabilities = np.clip(
        probabilities,
        1e-6,
        1 - 1e-6
    )

    logits = np.log(
        probabilities /
        (1 - probabilities)
    )

    calibrated = calibrator.predict_proba(
        logits.reshape(-1, 1)
    )[:, 1]

    return calibrated


def calculate_ece(
    y_true,
    probabilities,
    n_bins=10
):

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    bins = np.linspace(
        0,
        1,
        n_bins + 1
    )

    ece = 0.0

    for i in range(n_bins):

        if i == n_bins - 1:

            mask = (
                (probabilities >= bins[i]) &
                (probabilities <= bins[i + 1])
            )

        else:

            mask = (
                (probabilities >= bins[i]) &
                (probabilities < bins[i + 1])
            )

        if mask.sum() == 0:
            continue

        confidence = probabilities[mask].mean()
        accuracy = y_true[mask].mean()

        ece += (
            mask.mean() *
            abs(confidence - accuracy)
        )

    return ece


def create_risk_buckets(
    y_true,
    probabilities,
    n_buckets=10
):

    result = pd.DataFrame({
        "actual": np.asarray(y_true),
        "probability": np.asarray(probabilities)
    })

    result["risk_bucket"] = pd.qcut(
        result["probability"],
        q=n_buckets,
        labels=False,
        duplicates="drop"
    )

    buckets = (
        result
        .groupby(
            "risk_bucket",
            observed=True
        )
        .agg(
            observations=("actual", "size"),
            mean_probability=("probability", "mean"),
            actual_rate=("actual", "mean")
        )
        .reset_index()
    )

    buckets["risk_bucket"] = (
        buckets["risk_bucket"] + 1
    )

    return buckets


def top_risk_analysis(
    y_true,
    probabilities
):

    result = pd.DataFrame({
        "actual": np.asarray(y_true),
        "probability": np.asarray(probabilities)
    })

    result = result.sort_values(
        "probability",
        ascending=False
    )

    baseline_rate = result["actual"].mean()

    rows = []

    for percentage in [
        0.1,
        0.5,
        1,
        2,
        5,
        10
    ]:

        n = max(
            1,
            int(len(result) * percentage / 100)
        )

        top = result.head(n)

        rate = top["actual"].mean()

        lift = (
            rate / baseline_rate
            if baseline_rate > 0
            else np.nan
        )

        rows.append({
            "top_percent": percentage,
            "observations": n,
            "adverse_rate": rate,
            "baseline_rate": baseline_rate,
            "lift": lift
        })

    return pd.DataFrame(rows)


def main():

    all_results = []
    all_buckets = []
    all_top_risk = []

    for symbol in STOCKS:

        print("\n" + "=" * 70)
        print(f"STOCK: {symbol}")
        print("=" * 70)

        df = prepare_data(symbol)

        X = df[FEATURE_COLUMNS]
        y = df["adverse_buy"]

        n = len(df)

        train_end = int(n * 0.60)
        calibration_end = int(n * 0.80)

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]

        X_calibration = X.iloc[
            train_end:calibration_end
        ]

        y_calibration = y.iloc[
            train_end:calibration_end
        ]

        X_test = X.iloc[
            calibration_end:
        ]

        y_test = y.iloc[
            calibration_end:
        ]

        print(
            "Train:",
            len(X_train),
            "Calibration:",
            len(X_calibration),
            "Test:",
            len(X_test)
        )

        model = train_xgboost(
            X_train,
            y_train
        )

        calibration_probabilities = (
            model.predict_proba(
                X_calibration
            )[:, 1]
        )

        test_probabilities = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        calibrator = calibrate_probabilities(
            calibration_probabilities,
            y_calibration
        )

        calibrated_probabilities = apply_calibration(
            calibrator,
            test_probabilities
        )

        raw_auc = roc_auc_score(
            y_test,
            test_probabilities
        )

        calibrated_auc = roc_auc_score(
            y_test,
            calibrated_probabilities
        )

        raw_pr_auc = average_precision_score(
            y_test,
            test_probabilities
        )

        calibrated_pr_auc = average_precision_score(
            y_test,
            calibrated_probabilities
        )

        raw_brier = brier_score_loss(
            y_test,
            test_probabilities
        )

        calibrated_brier = brier_score_loss(
            y_test,
            calibrated_probabilities
        )

        raw_ece = calculate_ece(
            y_test,
            test_probabilities
        )

        calibrated_ece = calculate_ece(
            y_test,
            calibrated_probabilities
        )

        print(
            f"Raw ROC-AUC:        {raw_auc:.4f}"
        )

        print(
            f"Calibrated ROC-AUC: {calibrated_auc:.4f}"
        )

        print(
            f"Raw PR-AUC:         {raw_pr_auc:.4f}"
        )

        print(
            f"Calibrated PR-AUC:  {calibrated_pr_auc:.4f}"
        )

        print(
            f"Raw Brier:          {raw_brier:.6f}"
        )

        print(
            f"Calibrated Brier:   {calibrated_brier:.6f}"
        )

        print(
            f"Raw ECE:            {raw_ece:.6f}"
        )

        print(
            f"Calibrated ECE:     {calibrated_ece:.6f}"
        )

        buckets = create_risk_buckets(
            y_test,
            calibrated_probabilities
        )

        buckets.insert(
            0,
            "stock",
            symbol
        )

        all_buckets.append(
            buckets
        )

        top_risk = top_risk_analysis(
            y_test,
            calibrated_probabilities
        )

        top_risk.insert(
            0,
            "stock",
            symbol
        )

        all_top_risk.append(
            top_risk
        )

        print("\nRisk buckets:")
        print(buckets)

        all_results.append({
            "stock": symbol,
            "raw_auc": raw_auc,
            "calibrated_auc": calibrated_auc,
            "raw_pr_auc": raw_pr_auc,
            "calibrated_pr_auc": calibrated_pr_auc,
            "raw_brier": raw_brier,
            "calibrated_brier": calibrated_brier,
            "raw_ece": raw_ece,
            "calibrated_ece": calibrated_ece
        })

    results = pd.DataFrame(
        all_results
    )

    buckets = pd.concat(
        all_buckets,
        ignore_index=True
    )

    top_risk = pd.concat(
        all_top_risk,
        ignore_index=True
    )

    print("\n" + "=" * 70)
    print("FINAL CALIBRATION SUMMARY")
    print("=" * 70)

    print(
        results[
            [
                "raw_auc",
                "calibrated_auc",
                "raw_pr_auc",
                "calibrated_pr_auc",
                "raw_brier",
                "calibrated_brier",
                "raw_ece",
                "calibrated_ece"
            ]
        ].mean()
    )

    results.to_csv(
        "results/calibration_results.csv",
        index=False
    )

    buckets.to_csv(
        "results/risk_buckets.csv",
        index=False
    )

    top_risk.to_csv(
        "results/risk_concentration.csv",
        index=False
    )

    plot_data = []

    for _, row in buckets.iterrows():

        plot_data.append({
            "stock": row["stock"],
            "predicted": row["mean_probability"],
            "actual": row["actual_rate"]
        })

    plot_data = pd.DataFrame(
        plot_data
    )

    grouped = (
        plot_data
        .groupby("stock")
        .mean()
    )

    plt.figure(figsize=(8, 8))

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    plt.scatter(
        grouped["predicted"],
        grouped["actual"]
    )

    plt.xlabel(
        "Mean predicted probability"
    )

    plt.ylabel(
        "Observed adverse-event rate"
    )

    plt.title(
        "Calibration Across Stocks"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "plots/calibration_curve.png",
        dpi=200
    )

    plt.close()

    concentration = (
        top_risk
        .groupby("top_percent")
        .agg(
            adverse_rate=("adverse_rate", "mean"),
            lift=("lift", "mean")
        )
        .reset_index()
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        concentration["top_percent"],
        concentration["lift"],
        marker="o"
    )

    plt.xlabel(
        "Highest-risk observations retained (%)"
    )

    plt.ylabel(
        "Average adverse-event lift"
    )

    plt.title(
        "Adverse-Event Concentration by Predicted Risk"
    )

    plt.xscale("log")

    plt.tight_layout()

    plt.savefig(
        "plots/risk_concentration.png",
        dpi=200
    )

    plt.close()

    bucket_plot = (
        buckets
        .groupby("risk_bucket")
        .agg(
            predicted=(
                "mean_probability",
                "mean"
            ),
            actual=(
                "actual_rate",
                "mean"
            )
        )
        .reset_index()
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        bucket_plot["risk_bucket"],
        bucket_plot["predicted"],
        marker="o",
        label="Predicted"
    )

    plt.plot(
        bucket_plot["risk_bucket"],
        bucket_plot["actual"],
        marker="o",
        label="Observed"
    )

    plt.xlabel(
        "Risk bucket"
    )

    plt.ylabel(
        "Adverse-event probability"
    )

    plt.title(
        "Predicted vs Observed Risk"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "plots/risk_buckets.png",
        dpi=200
    )

    plt.close()

    print("\nSaved:")
    print("results/calibration_results.csv")
    print("results/risk_buckets.csv")
    print("results/risk_concentration.csv")
    print("plots/calibration_curve.png")
    print("plots/risk_concentration.png")
    print("plots/risk_buckets.png")


if __name__ == "__main__":
    main()