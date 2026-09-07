import numpy as np

from src.database import get_stock_data
from src.features import create_features
from src.labels import create_labels
from src.model import train_model, evaluate_model


symbol = "AXISBANK"

df = get_stock_data(symbol)

print("Raw data:")
print(df.head())

print("\nShape:")
print(df.shape)

df["timestamp"] = df["timestamp"].astype("datetime64[ns]")

print("\nDuplicate timestamps:", df["timestamp"].duplicated().sum())

print("\nTimestamp differences:")
print(df["timestamp"].diff().value_counts().head(20))


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


df = create_labels(
    df,
    horizon_seconds=10,
    threshold=0.001
)


df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()


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


X = df[feature_columns]

y = df["adverse_buy"]


print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentage:")
print(y.value_counts(normalize=True))


split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]


print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


model = train_model(
    X_train,
    y_train
)


probabilities = evaluate_model(
    model,
    X_test,
    y_test
)