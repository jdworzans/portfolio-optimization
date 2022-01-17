import pandas as pd

BASE_URL = "https://stooq.com/q/d/l/?s={ticker}"
WIG20_TICKERS = ["ACP", "ALE", "CCC", "CDR", "CPS", "DNP", "JSW", "KGH", "LPP", "LTS", "MRC", "OPL", "PEO", "PGE", "PGN", "PKN", "PKO", "PZU", "SPL", "TPE"]

def download_historical_data(ticker) -> pd.DataFrame:
    data_url = BASE_URL.format(ticker=ticker)
    return pd.read_csv(data_url)


for ticker in WIG20_TICKERS:
    download_historical_data(ticker).to_csv(f"data/raw/{ticker}.csv")
