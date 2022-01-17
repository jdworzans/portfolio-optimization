from pathlib import Path

import pandas as pd

RAW_DATASETS_DIR = Path("data/raw")
PREPROCESSED_DATASETS_DIR = Path("data/preprocessed")
PREPROCESSED_DATASETS_DIR.mkdir(exist_ok=True)

def preprocess(df):
    df["Date"] = pd.to_datetime(df["Date"])
    df["Change"] = df["Close"].pct_change()
    df = df[["Date", "Change"]].set_index("Date")
    return df

for dataset_filepath in RAW_DATASETS_DIR.iterdir():
    df = pd.read_csv(dataset_filepath, index_col=0)
    preprocess(df).to_csv(PREPROCESSED_DATASETS_DIR / dataset_filepath.name)
