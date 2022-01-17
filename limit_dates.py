from pathlib import Path

import pandas as pd

PREPROCESSED_DATASETS_DIR = Path("data/preprocessed")
LIMITED_DATASETS_DIR = Path("data/limited")

LIMITED_DATASETS_DIR.mkdir(exist_ok=True)

dfs = {}
min_date = pd.Timestamp("1900-01-01")
max_date = pd.Timestamp("2022-01-01")

for filepath in PREPROCESSED_DATASETS_DIR.iterdir():
    dfs[filepath] = pd.read_csv(filepath, parse_dates=["Date"])
    # print(dfs[filepath])

    min_date = max(min_date, dfs[filepath]["Date"].min())
    max_date = min(max_date, dfs[filepath]["Date"].max())

min_date += pd.Timedelta(days=1)

for filepath, df in dfs.items():
    df = df[(df["Date"] >= min_date) & (df["Date"] <= max_date)]
    df.set_index("Date").to_csv(LIMITED_DATASETS_DIR / filepath.name)
