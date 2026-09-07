import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

from src.database import get_stock_data
from src.features import create_features
from src.labels import create_labels
from src.model import train_model


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
        print("Not enough usable data")
        return None

    print(
        "Observations:",
        len(df)
    )

    print(
        "Adverse rate:",
        f"{df['adverse_buy'].mean():.4%}"
    )

    return df


print(
    "=============================="
)

print(
    "CROSS-STOCK VALIDATION"
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


available_stocks = list(
    stock_data.keys()
)


print(
    "\n=============================="
)

print(
    "AVAILABLE STOCKS"
)

print(
    "=============================="
)


for stock in available_stocks:

    print(
        stock,
        len(stock_data[stock])
    )


if len(available_stocks) < 3:

    raise ValueError(
        "At least 3 stocks are required."
    )


overall_results = []
precision_results = []


for test_stock in available_stocks:

    print(
        "\n=============================="
    )

    print(
        "TEST STOCK:",
        test_stock
    )

    print(
        "=============================="
    )

    train_stocks = [
        stock
        for stock in available_stocks
        if stock != test_stock
    ]

    train_data = pd.concat(
        [
            stock_data[stock]
            for stock in train_stocks
        ],
        ignore_index=True
    )

    test_data = stock_data[
        test_stock
    ].copy()

    X_train = train_data[
        FEATURE_COLUMNS
    ]

    y_train = train_data[
        "adverse_buy"
    ]

    X_test = test_data[
        FEATURE_COLUMNS
    ]

    y_test = test_data[
        "adverse_buy"
    ]

    print(
        "Training rows:",
        len(X_train)
    )

    print(
        "Testing rows:",
        len(X_test)
    )

    model = train_model(
        X_train,
        y_train
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    baseline = y_test.mean()

    print(
        "\nROC-AUC:",
        f"{auc:.4f}"
    )

    print(
        "Baseline adverse rate:",
        f"{baseline:.4%}"
    )

    row = {
        "test_stock": test_stock,
        "training_stocks": len(train_stocks),
        "test_observations": len(test_data),
        "roc_auc": auc,
        "baseline_adverse_rate": baseline
    }

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

        row[
            f"top_{key}_precision"
        ] = precision

        row[
            f"top_{key}_lift"
        ] = lift

        precision_results.append({
            "test_stock": test_stock,
            "top_percentage": percentage,
            "observations": n,
            "adverse_events": int(
                actual.sum()
            ),
            "precision": precision,
            "lift": lift
        })

        print(
            f"Top {percentage}%:"
            f" precision={precision:.4f}"
            f" lift={lift:.2f}x"
        )

    overall_results.append(row)


results_df = pd.DataFrame(
    overall_results
)

precision_df = pd.DataFrame(
    precision_results
)


print(
    "\n=============================="
)

print(
    "FINAL CROSS-STOCK RESULTS"
)

print(
    "=============================="
)


print(
    results_df[
        [
            "test_stock",
            "roc_auc",
            "baseline_adverse_rate",
            "top_0_5_precision",
            "top_0_5_lift",
            "top_1_precision",
            "top_1_lift",
            "top_2_precision",
            "top_2_lift"
        ]
    ].to_string(index=False)
)


print(
    "\n=============================="
)

print(
    "SUMMARY"
)

print(
    "=============================="
)


print(
    "Mean ROC-AUC:",
    f"{results_df['roc_auc'].mean():.4f}"
)

print(
    "Median ROC-AUC:",
    f"{results_df['roc_auc'].median():.4f}"
)

print(
    "Minimum ROC-AUC:",
    f"{results_df['roc_auc'].min():.4f}"
)

print(
    "Maximum ROC-AUC:",
    f"{results_df['roc_auc'].max():.4f}"
)

print(
    "\nMean Top 0.5% Precision:",
    f"{results_df['top_0_5_precision'].mean():.4%}"
)

print(
    "Mean Top 0.5% Lift:",
    f"{results_df['top_0_5_lift'].mean():.2f}x"
)

print(
    "\nMean Top 1% Precision:",
    f"{results_df['top_1_precision'].mean():.4%}"
)

print(
    "Mean Top 1% Lift:",
    f"{results_df['top_1_lift'].mean():.2f}x"
)


results_df.to_csv(
    "experiments/cross_stock_results.csv",
    index=False
)

precision_df.to_csv(
    "experiments/cross_stock_precision.csv",
    index=False
)


print(
    "\nSaved:"
)

print(
    "experiments/cross_stock_results.csv"
)

print(
    "experiments/cross_stock_precision.csv"
)

print(
    "\nValidation complete."
)