import argparse
import logging
from pathlib import Path

import pandas as pd

RAW_DATASETS_DIR = Path("data/raw")
PREPROCESSED_DATASETS_DIR = Path("data/preprocessed")
RAW_DATASETS_DIR.mkdir(exist_ok=True)
PREPROCESSED_DATASETS_DIR.mkdir(exist_ok=True)
DEFAULT_MIN_DATE = pd.Timestamp("1900-01-01")
DEFAULT_MAX_DATE = pd.Timestamp.today()

BASE_URL = "https://stooq.com/q/d/l/?s={ticker}"
TICKERS = {
    # stocks from WIG20 (30.04.2009) which are still available
    "WIG20": [
        "PKO",
        "PEO",
        "OPL",
        "PKN",
        "KGH",
        "PGN",
        "ACP",
        "GTC",
        "SPL",
        "PBG",
        "CEZ",
        "GTN",
        "PXM",
        "MBK",
        "CPS",
        "LTS",
        "AGO",
        "BIO",
    ],
    "WIG-BANKI": [
        "ALR",
        "BNP",
        "BOS",
        "GTN",
        "GNB",
        "BHW",
        "ING",
        "MBK",
        "MIL",
        "PEO",
        "PKO",
        "SPL",
        "SAN",
        "UCG",
    ],
}


def download(ticker: str) -> pd.DataFrame:
    data_url = BASE_URL.format(ticker=ticker)
    return pd.read_csv(data_url)


def is_downloaded(ticker: str) -> bool:
    return (RAW_DATASETS_DIR / f"{ticker}.csv").is_file()


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"])
    df["Change"] = df["Close"].pct_change()
    df = df[["Date", "Change"]].set_index("Date")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--ticker", action="append", default=[])
    parser.add_argument("-i", "--index", action="append", default=[])
    parser.add_argument("-n", "--name", default="dataset")
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Prepare list of all considered tickers
    tickers = set(args.ticker)
    for index in args.index:
        if index in TICKERS:
            tickers.update(TICKERS[index])
        else:
            logging.warn(f"Index {index} not known.")

    # Download stock data for all tickers if not available
    for ticker in tickers:
        if not is_downloaded(ticker) or args.force:
            logging.info(f"{ticker} not found. Downloading...")
            df = download(ticker)
            df.to_csv(RAW_DATASETS_DIR / f"{ticker}.csv")
        else:
            logging.info(f"{ticker} found. Skip.")

    dfs = {}
    min_date = DEFAULT_MIN_DATE
    max_date = DEFAULT_MAX_DATE
    for ticker in tickers:
        df = pd.read_csv(RAW_DATASETS_DIR / f"{ticker}.csv")
        df = preprocess(df)
        dfs[ticker] = df

        # Find minimum and maximum date across the data
        min_date = max(min_date, df.index.min())
        max_date = min(max_date, df.index.max())
    min_date += pd.Timedelta(days=1)

    # Limit dates
    for ticker in dfs:
        dfs[ticker] = dfs[ticker][str(min_date):str(max_date)]

    # Create single final table
    final_df = pd.DataFrame()

    for ticker, df in dfs.items():
        final_df[ticker] = df["Change"]

    # Drop all nans
    final_df = final_df.dropna()
    final_df.to_csv(Path("data") / f"{args.name}.csv")
