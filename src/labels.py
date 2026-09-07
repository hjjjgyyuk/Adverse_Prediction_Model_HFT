import numpy as np


def create_labels(df, horizon_seconds=5, threshold=0.001):

    df = df.copy()

    future_price = df["ltp"].shift(-horizon_seconds)

    future_return = (
        future_price / df["ltp"] - 1
    )

    df["future_return"] = future_return

    valid = (
        df["ltp"].notna() &
        future_price.notna()
    )

    df["adverse_buy"] = np.nan
    df["adverse_sell"] = np.nan

    df.loc[valid, "adverse_buy"] = (
        future_return[valid] <= -threshold
    ).astype(int)

    df.loc[valid, "adverse_sell"] = (
        future_return[valid] >= threshold
    ).astype(int)

    return df