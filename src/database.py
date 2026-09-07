import pandas as pd
from sqlalchemy import create_engine


DATABASE_URL = (
    "postgresql+psycopg2://postgres:ENTER_PASSWORD"
    "@localhost:5432/hft-stock"
)

engine = create_engine(DATABASE_URL)


def get_stock_data(symbol):
    query = """
        SELECT timestamp, ltp, volume
        FROM stock_ticks
        WHERE symbol = %(symbol)s
        ORDER BY timestamp
    """

    df = pd.read_sql(
        query,
        engine,
        params={"symbol": symbol}
    )

    return df