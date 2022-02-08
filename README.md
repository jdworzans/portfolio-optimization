
## Przed uruchomieniem:
Należy zainstalować implementację `nsga2` oraz biblioteki zawarte w `requirements.txt`:
```
pip install -e nsga2
pip install -r requirements.txt
```

## Pobieranie danych:
Aby pobrać dane dotyczące konkretnego indeksu, należy wykonać polecenie (np. dla WIG20):
```
python3 preprocessing/prepare_data.py -i WIG20
```
Uwaga: Dla wszystkich notowań (`-i GPW`)operacja może nie udać się podczas pojedynczego uruchomienia
ze względu na limit pobierania notowań. Aby uzupełnić brakujące dane należy usunąć wadliwe pliki
i ponownie uruchomić polecenie po upłynięciu odpowiedniego czasu (lub z wykorzystaniem proxy itp.).

## Uruchamianie testów dla pakietu `nsga2`:
```
pip install -e nsga2
pytest nsga2
```
