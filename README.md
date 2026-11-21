# ZAB – Zadanie 1 (wersja A): imputacja danych

Rozwiązanie zadania z przedmiotu Zaawansowana analityka biznesowa – metody imputacji danych.

## Struktura repozytorium

```
zab-zadanie1/
├── data/                        # zbiory danych
│   └── ka01.csv                 # indywidualny zbiór grupy
├── notebooks/
│   └── 01_analiza_struktury.ipynb   # PUNKT 1 – analiza struktury i braków
├── figures/                     # wykresy generowane przez notebooki
├── src/                         # (opcjonalnie) wspólne funkcje grupy
├── requirements.txt
├── .gitignore
└── README.md
```

## Uruchomienie

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab                      # otwórz notebooks/01_analiza_struktury.ipynb
```
