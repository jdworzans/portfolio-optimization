## Reproducing data
```
python3 preprocessing/download.py
python3 preprocessing/preprocess.py
python3 preprocessing/limit_dates.py
python3 preprocessing/merge.py
```

## Running tests
```
pip install -e nsga2
pytest nsga2
```