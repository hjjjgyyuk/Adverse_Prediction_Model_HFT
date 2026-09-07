import numpy as np


def create_features(df):

    df = df.copy()

    df["return_1s"] = df["ltp"].pct_change(fill_method=None)
    df["return_2s"] = df["ltp"].pct_change(fill_method=None)
    df["return_5s"] = df["ltp"].pct_change(fill_method=None)
    df["return_10s"] = df["ltp"].pct_change(fill_method=None)
    df["return_20s"] = df["ltp"].pct_change(fill_method=None)
    df["return_30s"] = df["ltp"].pct_change(fill_method=None)
    df["return_60s"] = df["ltp"].pct_change(fill_method=None)

    df["momentum_acceleration"] = (
        df["return_5s"] - df["return_20s"]
    )

    df["volatility_10s"] = (
        df["return_1s"].rolling(10).std()
    )

    df["volatility_30s"] = (
        df["return_1s"].rolling(30).std()
    )

    df["volatility_60s"] = (
        df["return_1s"].rolling(60).std()
    )

    df["volume_change"] = df["volume"].diff()

    df["volume_10s"] = (
        df["volume_change"].rolling(10).sum()
    )

    df["volume_30s"] = (
        df["volume_change"].rolling(30).sum()
    )

    df["volume_change_pct"] = (
        df["volume"].pct_change(fill_method=None)
    )

    df["up_moves_10s"] = (
        (df["return_1s"] > 0).rolling(10).sum()
    )

    df["down_moves_10s"] = (
        (df["return_1s"] < 0).rolling(10).sum()
    )

    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute

    df = df.replace([np.inf, -np.inf], np.nan)

    return df