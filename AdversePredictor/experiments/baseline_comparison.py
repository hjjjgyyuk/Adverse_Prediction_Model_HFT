import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier

from src.database import get_stock_data
from src.features import create_features
from src.labels import create_labels


HORIZON = 5
THRESHOLD = 0.001


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


VOLATILITY_FEATURES = [
    "volatility_10s",
    "volatility_30s",
    "volatility_60s"
]


VOLUME_FEATURES = [
    "volume_change",
    "volume_10s",
    "volume_30s",
    "volume_change_pct"
]


def prepare_stock(symbol):

    print(
        "\nLoading:",
        symbol
    )

    df = get_stock_data(symbol)

    if len(df) == 0:
        print("No data found")
        return None

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

    df = df.dropna(
        subset=FEATURE_COLUMNS + [
            "adverse_buy"
        ]
    )

    if len(df) < 5000:
        print("Not enough usable observations")
        return None

    print(
        "Observations:",
        len(df)
    )

    return df


def train_xgboost(
    X_train,
    y_train
):

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


def evaluate_model(
    name,
    probabilities,
    y_test,
    stock
):

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    baseline = y_test.mean()

    result = {
        "stock": stock,
        "model": name,
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "adverse_rate": baseline
    }

    print(
        f"\n{name}"
    )

    print(
        "ROC-AUC:",
        f"{auc:.4f}"
    )

    print(
        "PR-AUC:",
        f"{pr_auc:.4f}"
    )

    print(
        "Adverse rate:",
        f"{baseline:.4%}"
    )

    for percentage in [
        0.5,
        1,
        2,
        5
    ]:

        n = max(
            1,
            int(
                len(probabilities)
                * percentage
                / 100
            )
        )

        indices = np.argsort(
            probabilities
        )[-n:]

        actual = y_test.iloc[
            indices
        ]

        precision = actual.mean()

        lift = (
            precision / baseline
            if baseline > 0
            else np.nan
        )

        key = (
            str(percentage)
            .replace(".", "_")
        )

        result[
            f"top_{key}_precision"
        ] = precision

        result[
            f"top_{key}_lift"
        ] = lift

        print(
            f"Top {percentage}%:"
            f" precision={precision:.4f}"
            f" lift={lift:.2f}x"
        )

    return result


print(
    "=============================="
)

print(
    "BASELINE COMPARISON"
)

print(
    "=============================="
)

print(
    "\nHorizon:",
    HORIZON,
    "seconds"
)

print(
    "Adverse threshold:",
    THRESHOLD * 100,
    "%"
)


stock_data = {}


for stock in STOCKS:

    data = prepare_stock(stock)

    if data is not None:
        stock_data[stock] = data


results = []


for stock, df in stock_data.items():

    print(
        "\n=============================="
    )

    print(
        "STOCK:",
        stock
    )

    print(
        "=============================="
    )

    split = int(
        len(df) * 0.8
    )

    train_df = df.iloc[
        :split
    ].copy()

    test_df = df.iloc[
        split:
    ].copy()

    y_train = train_df[
        "adverse_buy"
    ]

    y_test = test_df[
        "adverse_buy"
    ]

    print(
        "Training:",
        len(train_df)
    )

    print(
        "Testing:",
        len(test_df)
    )

    print(
        "Training period:",
        train_df["timestamp"].min(),
        "to",
        train_df["timestamp"].max()
    )

    print(
        "Testing period:",
        test_df["timestamp"].min(),
        "to",
        test_df["timestamp"].max()
    )


    volatility_score = (
        test_df[
            VOLATILITY_FEATURES
        ]
        .mean(axis=1)
    )

    result = evaluate_model(
        "Volatility",
        volatility_score,
        y_test,
        stock
    )

    results.append(result)


    volume_score = (
        test_df[
            VOLUME_FEATURES
        ]
        .abs()
        .mean(axis=1)
    )

    result = evaluate_model(
        "Volume",
        volume_score,
        y_test,
        stock
    )

    results.append(result)


    X_train = train_df[
        FEATURE_COLUMNS
    ]

    X_test = test_df[
        FEATURE_COLUMNS
    ]


    logistic = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    logistic.fit(
        X_train,
        y_train
    )

    probabilities = logistic.predict_proba(
        X_test
    )[:, 1]

    result = evaluate_model(
        "Logistic Regression",
        probabilities,
        y_test,
        stock
    )

    results.append(result)


    xgb = train_xgboost(
        X_train,
        y_train
    )

    probabilities = xgb.predict_proba(
        X_test
    )[:, 1]

    result = evaluate_model(
        "XGBoost",
        probabilities,
        y_test,
        stock
    )

    results.append(result)


results_df = pd.DataFrame(
    results
)


print(
    "\n=============================="
)

print(
    "FINAL COMPARISON"
)

print(
    "=============================="
)


summary_columns = [
    "stock",
    "model",
    "roc_auc",
    "pr_auc",
    "adverse_rate",
    "top_0_5_precision",
    "top_0_5_lift",
    "top_1_precision",
    "top_1_lift"
]


print(
    results_df[
        summary_columns
    ].to_string(
        index=False
    )
)


print(
    "\n=============================="
)

print(
    "MODEL SUMMARY"
)

print(
    "=============================="
)


model_summary = (
    results_df
    .groupby("model")
    .agg({
        "roc_auc": [
            "mean",
            "median"
        ],
        "pr_auc": [
            "mean",
            "median"
        ],
        "top_0_5_precision": "mean",
        "top_0_5_lift": "mean",
        "top_1_precision": "mean",
        "top_1_lift": "mean"
    })
)


print(
    model_summary
)


print(
    "\n=============================="
)

print(
    "XGBOOST ADVANTAGE"
)

print(
    "=============================="
)


pivot = results_df.pivot(
    index="stock",
    columns="model",
    values="roc_auc"
)


if (
    "XGBoost" in pivot.columns
    and
    "Logistic Regression" in pivot.columns
):

    difference = (
        pivot["XGBoost"]
        -
        pivot["Logistic Regression"]
    )

    print(
        "\nXGBoost ROC-AUC minus"
        " Logistic Regression:"
    )

    print(
        difference
    )

    print(
        "\nMean difference:",
        f"{difference.mean():.4f}"
    )


results_df.to_csv(
    "experiments/baseline_results.csv",
    index=False
)


print(
    "\nSaved:"
)

print(
    "experiments/baseline_results.csv"
)

print(
    "\nBaseline comparison complete."
)