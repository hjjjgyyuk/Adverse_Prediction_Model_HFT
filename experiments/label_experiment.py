import numpy as np

from src.database import get_stock_data
from src.features import create_features
from src.labels import create_labels
from src.model import train_model

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score
)


symbol = "AXISBANK"

df = get_stock_data(symbol)

df["timestamp"] = df["timestamp"].astype("datetime64[ns]")

df = (
    df.set_index("timestamp")
      .resample("1s")
      .agg({
          "ltp": "last",
          "volume": "last"
      })
      .dropna()
      .reset_index()
)

df = create_features(df)


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


horizon = 5
threshold = 0.001

df = create_labels(
    df,
    horizon_seconds=horizon,
    threshold=threshold
)

df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna()


X = df[feature_columns]
y = df["adverse_buy"]


split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]


print("Horizon:", horizon, "seconds")
print("Threshold:", threshold * 100, "%")

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentage:")
print(y.value_counts(normalize=True))


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

print("\nROC-AUC:", auc)


print("\nProbability statistics:")
print("Minimum:", probabilities.min())
print("Maximum:", probabilities.max())
print("Mean:", probabilities.mean())


percentiles = [
    50,
    75,
    90,
    95,
    99,
    99.5,
    99.9
]

print("\nProbability percentiles:")

for percentile in percentiles:

    value = np.percentile(
        probabilities,
        percentile
    )

    print(
        f"{percentile}%: {value:.6f}"
    )


print("\nThreshold evaluation:")

thresholds = [
    0.01,
    0.02,
    0.03,
    0.05,
    0.08,
    0.10,
    0.15,
    0.20
]

print(
    f"{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'Predicted':<12}"
)


for threshold_value in thresholds:

    predictions = (
        probabilities >= threshold_value
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    predicted = predictions.sum()

    print(
        f"{threshold_value:<12.2f}"
        f"{precision:<12.3f}"
        f"{recall:<12.3f}"
        f"{predicted:<12}"
    )


print("\nPrecision at top percentage of predictions:")

print(
    f"{'Top %':<10}"
    f"{'Observations':<15}"
    f"{'Adverse':<12}"
    f"{'Precision':<12}"
)


for percentage in [0.5, 1, 2, 5, 10]:

    n = max(
        1,
        int(len(probabilities) * percentage / 100)
    )

    top_indices = np.argsort(
        probabilities
    )[-n:]

    top_actual = y_test.iloc[top_indices]

    precision = top_actual.mean()

    print(
        f"{percentage:<10.1f}"
        f"{n:<15}"
        f"{top_actual.sum():<12}"
        f"{precision:<12.4f}"
    )