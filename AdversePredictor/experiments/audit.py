import pandas as pd

from src.database import get_stock_data


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


for symbol in STOCKS:

    print("\n" + "=" * 70)
    print(symbol)
    print("=" * 70)

    df = get_stock_data(symbol)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    )

    print(
        "Raw rows:",
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

    print(
        "Duplicate timestamps:",
        df["timestamp"].duplicated().sum()
    )

    gaps = (
        df["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    print("\nTimestamp gap statistics:")

    print(
        gaps.describe()
    )

    print("\nLargest gaps:")

    print(
        gaps.nlargest(10)
    )

    print("\nGaps greater than 60 seconds:")

    large_gaps = gaps[gaps > 60]

    print(
        "Count:",
        len(large_gaps)
    )

    print(
        large_gaps.describe()
    )

    print("\nAfter 1-second resampling:")

    resampled = (
        df.set_index("timestamp")
        .resample("1s")
        .last()
        .ffill()
        .reset_index()
    )

    print(
        "Rows:",
        len(resampled)
    )

    print(
        "Start:",
        resampled["timestamp"].min()
    )

    print(
        "End:",
        resampled["timestamp"].max()
    )