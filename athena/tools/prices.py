import duckdb
import yfinance as yf

DB_PATH = "data/athena.duckdb"


def download_prices(ticker, start="2022-01-01"):
    df = yf.download(ticker, start=start, auto_adjust=True,
                     multi_level_index=False, progress=False)
    df = df.reset_index()
    df["ticker"] = ticker
    df = df[["ticker", "Date", "Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["ticker", "date", "open", "high", "low", "close", "volume"]

    con = duckdb.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            ticker VARCHAR, date DATE, open DOUBLE, high DOUBLE,
            low DOUBLE, close DOUBLE, volume DOUBLE
        )
    """)
    con.execute("DELETE FROM prices WHERE ticker = ?", [ticker])
    con.execute("INSERT INTO prices SELECT * FROM df")
    con.close()
    return len(df)


def get_prices(ticker, as_of):
    con = duckdb.connect(DB_PATH)
    df = con.execute(
        "SELECT * FROM prices WHERE ticker = ? AND date <= ? ORDER BY date",
        [ticker, as_of],
    ).df()
    con.close()
    return df