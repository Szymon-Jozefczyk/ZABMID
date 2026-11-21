# ZAB – Zadanie 1 (wersja A): imputacja danych

Rozwiązanie zadania z przedmiotu *Zaawansowana analityka biznesowa – metody imputacji danych* (SGH).
Cel: ocena średniej zmiany `CHG = Y_T2 − Y_T1` w grupie z czynnikiem (`TRT=1`) i kontrolnej (`TRT=2`)
modelem analizy kowariancji `chg = β0 + β1·TRT + β2·Y_T1 + ε`.

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
jupyter lab                      # i otwórz notebooks/01_analiza_struktury.ipynb
```

Notebooki wykrywają korzeń projektu automatycznie, więc działają po sklonowaniu bez zmiany ścieżek.

## Podział na punkty zadania

| Punkt | Plik | Autor |
|------|------|-------|
| 1. Analiza struktury i braków | `notebooks/01_analiza_struktury.ipynb` | — |
| 2. Ocena mechanizmu braków | `notebooks/02_mechanizm.ipynb` | — |
| 3. Estymacja (MCAR/MAR) + tipping-point | `notebooks/03_estymacja.ipynb` | — |
| 4. Model właściwy | `notebooks/04_model_wlasciwy.ipynb` | — |
| 5. Miara diagnostyczna | `notebooks/05_diagnostyka.ipynb` | — |
