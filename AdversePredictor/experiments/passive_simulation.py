import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.database import get_stock_data
from src.features import create_features
from src.labels import create_labels
from src.model import train_model


symbol = "AXISBANK"

horizon = 5
label_threshold = 0.001


feature_columns = [
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


df = get_stock_data(symbol)

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
    horizon_seconds=horizon,
    threshold=label_threshold
)


df = df.replace(
    [np.inf, -np.inf],
    np.nan
)


df = df.dropna(
    subset=feature_columns + [
        "adverse_buy",
        "future_return"
    ]
)


X = df[
    feature_columns
]

y = df[
    "adverse_buy"
]


split = int(
    len(df) * 0.8
)


X_train = X.iloc[
    :split
]

X_test = X.iloc[
    split:
]


y_train = y.iloc[
    :split
]

y_test = y.iloc[
    split:
]


test_df = df.iloc[
    split:
].copy()


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


test_df["risk"] = probabilities


test_df["future_price"] = (
    test_df["ltp"]
    *
    (
        1
        +
        test_df["future_return"]
    )
)


test_df["gross_pnl"] = (
    test_df["future_price"]
    -
    test_df["ltp"]
)


test_df = test_df.dropna(
    subset=[
        "future_price",
        "gross_pnl"
    ]
)


def calculate_metrics(
    data,
    cost_per_trade=0
):

    if len(data) == 0:
        return None

    pnl = (
        data["gross_pnl"]
        -
        cost_per_trade
    )

    equity = pnl.cumsum()

    running_max = equity.cummax()

    drawdown = (
        equity
        -
        running_max
    )

    max_drawdown = drawdown.min()

    return {
        "fills": len(pnl),
        "total_pnl": pnl.sum(),
        "average_pnl": pnl.mean(),
        "median_pnl": pnl.median(),
        "win_rate": (
            pnl > 0
        ).mean(),
        "adverse_rate": (
            data["future_return"]
            <= -label_threshold
        ).mean(),
        "max_drawdown": max_drawdown
    }


print(
    "\n=============================="
)

print(
    "BASELINE"
)

print(
    "ALWAYS PROVIDE LIQUIDITY"
)

print(
    "=============================="
)


baseline = calculate_metrics(
    test_df,
    cost_per_trade=0
)


print(
    "Fills:",
    baseline["fills"]
)

print(
    "Total P&L:",
    baseline["total_pnl"]
)

print(
    "Average P&L:",
    baseline["average_pnl"]
)

print(
    "Median P&L:",
    baseline["median_pnl"]
)

print(
    "Win rate:",
    baseline["win_rate"]
)

print(
    "Adverse fill rate:",
    baseline["adverse_rate"]
)

print(
    "Max drawdown:",
    baseline["max_drawdown"]
)


print(
    "\n=============================="
)

print(
    "ML FILTERED LIQUIDITY"
)

print(
    "=============================="
)


risk_thresholds = [
    0.001,
    0.002,
    0.003,
    0.005,
    0.010,
    0.020
]


costs = [
    0.000,
    0.005,
    0.010,
    0.020
]


results = []


for cost in costs:

    print(
        "\n------------------------------"
    )

    print(
        "Transaction cost:",
        cost
    )

    print(
        "------------------------------"
    )

    print(
        f"{'Risk':<10}"
        f"{'Fills':<10}"
        f"{'Avg P&L':<12}"
        f"{'Total P&L':<15}"
        f"{'Win Rate':<12}"
        f"{'Adverse %':<12}"
        f"{'Max DD':<12}"
    )

    for risk_threshold in risk_thresholds:

        filtered = test_df[
            test_df["risk"]
            <
            risk_threshold
        ]

        if len(filtered) == 0:
            continue

        metrics = calculate_metrics(
            filtered,
            cost_per_trade=cost
        )

        results.append({
            "cost": cost,
            "risk_threshold": risk_threshold,
            **metrics
        })

        print(
            f"{risk_threshold:<10.3f}"
            f"{metrics['fills']:<10}"
            f"{metrics['average_pnl']:<12.5f}"
            f"{metrics['total_pnl']:<15.2f}"
            f"{metrics['win_rate']:<12.3f}"
            f"{metrics['adverse_rate']:<12.3f}"
            f"{metrics['max_drawdown']:<12.2f}"
        )


results_df = pd.DataFrame(
    results
)


results_df.to_csv(
    "experiments/simulation_results.csv",
    index=False
)


print(
    "\nResults saved to:"
)

print(
    "experiments/simulation_results.csv"
)


print(
    "\n=============================="
)

print(
    "RISK BUCKET ANALYSIS"
)

print(
    "=============================="
)


test_df["risk_bucket"] = pd.qcut(
    test_df["risk"],
    10,
    labels=False,
    duplicates="drop"
)


bucket_results = (
    test_df
    .groupby("risk_bucket")
    .agg(
        observations=(
            "gross_pnl",
            "count"
        ),
        average_risk=(
            "risk",
            "mean"
        ),
        average_pnl=(
            "gross_pnl",
            "mean"
        ),
        median_pnl=(
            "gross_pnl",
            "median"
        ),
        adverse_rate=(
            "future_return",
            lambda x: (
                x <= -label_threshold
            ).mean()
        )
    )
)


print(
    bucket_results
)


print(
    "\n=============================="
)

print(
    "EQUITY CURVES"
)

print(
    "=============================="
)


plt.figure(
    figsize=(12, 6)
)


baseline_pnl = test_df[
    "gross_pnl"
]

baseline_equity = (
    baseline_pnl.cumsum()
)


plt.plot(
    baseline_equity,
    label="Always provide"
)


for risk_threshold in [
    0.001,
    0.002,
    0.005
]:

    filtered = test_df[
        test_df["risk"]
        <
        risk_threshold
    ]

    equity = (
        filtered[
            "gross_pnl"
        ].cumsum()
    )

    plt.plot(
        equity,
        label=(
            f"ML filter < "
            f"{risk_threshold}"
        )
    )


plt.xlabel(
    "Trade / observation"
)

plt.ylabel(
    "Cumulative P&L"
)

plt.title(
    "Passive Liquidity Simulation"
)

plt.legend()

plt.tight_layout()


plt.savefig(
    "experiments/equity_curves.png",
    dpi=150
)

plt.show()


print(
    "\nEquity curve saved to:"
)

print(
    "experiments/equity_curves.png"
)

print(
    "\nSimulation complete."
)