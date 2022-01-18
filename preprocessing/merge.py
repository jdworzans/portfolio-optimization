from pathlib import Path

import pandas as pd

LIMITED_DATASETS_DIR = Path("data/limited")
MERGED_DATASET_FILEPATH = Path("data/dataset.csv")

df = pd.DataFrame()
dfs = {}

for filepath in LIMITED_DATASETS_DIR.iterdir():
    ticker = filepath.with_suffix("").name
    df[ticker] = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")["Change"]

df.to_csv(MERGED_DATASET_FILEPATH)
