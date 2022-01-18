from pathlib import Path

import pandas as pd

RAW_DATASETS_DIR = Path("data/raw")
BASE_URL = "https://stooq.com/q/d/l/?s={ticker}"
# stocks from WIG20 (30.04.2009) which are still available
TICKERS = ["PKO", "PEO", "OPL", "PKN", "KGH", "PGN", "ACP", "GTC", "SPL", "PBG", "CEZ", "GTN", "PXM", "MBK", "CPS", "LTS", "AGO", "BIO"]

RAW_DATASETS_DIR.mkdir(exist_ok=True)

def download_historical_data(ticker) -> pd.DataFrame:
    data_url = BASE_URL.format(ticker=ticker)
    return pd.read_csv(data_url)


for ticker in TICKERS:
    download_historical_data(ticker).to_csv(RAW_DATASETS_DIR / f"{ticker}.csv")
