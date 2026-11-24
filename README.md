# ZAB – Zadanie 1 (wersja A – zmienna ciągła): metody imputacji danych

Rozwiązanie zadania z przedmiotu *Zaawansowana analityka biznesowa – metody imputacji danych*.

## Struktura repozytorium

```text
ZABMID/
├── data/
│   └── ka01.csv                                       # zbiór danych grupy (N = 805)
├── notebooks/
│   ├── 00_raport_koncowy_ZAB_zadanie1A.ipynb          # zintegrowany raport (Pkt 1–3 + szablony Pkt 4–5)
│   ├── 01_analiza_struktury.ipynb                     # PUNKT 1 – analiza struktury i braków
│   ├── 02_analiza_mechanizmu_brakow.ipynb             # PUNKT 2 – ocena mechanizmu braków
│   └── 03_ancova_imputacja_mnar.ipynb                 # PUNKT 3 – ANCOVA (MCAR / MAR / MNAR 1D i 2D)
├── src/
│   └── punkt3_ancova_mi.py                            # moduł obliczeniowy i wykresy dla Punktu 3
├── figures/                                           # wykresy fig1–fig9 (300 dpi)
├── results/                                           # tabele CSV i raport/wydruk Punktu 3
├── kod_zadanie1A.py                                   # główny skrypt odtwarzający całe zadanie (MS Teams)
├── wydruk_wynikow_zadanie1A.txt                       # pełny wydruk wyników z konsoli (MS Teams)
├── raport_ZAB_zadanie1A.html                          # wyrenderowany raport końcowy (MS Teams)
├── requirements.txt
└── README.md
```


## Instrukcja dla pozostałych osób z zespołu (dokończenie Punktu 5)

Punkty **1, 2, 3 i 4** są ukończone. Do uzupełnienia pozostaje **Punkt 5**:

1. **Raport (HTML / PDF):**
   - Otwórz `notebooks/00_raport_koncowy_ZAB_zadanie1A.ipynb`.
   - Uzupełnij skład grupy na stronie tytułowej oraz wklej analizę, tabele, wykresy i wnioski w sekcji **Punkt 5**. Sekcja **Punkt 4** jest już uzupełniona.
   - Wygeneruj zaktualizowany plik HTML jednym poleceniem (z katalogu `ZABMID/`):
     ```powershell
     .\.venv\Scripts\python.exe -m jupyter nbconvert --to html --execute --output-dir . --output raport_ZAB_zadanie1A notebooks\00_raport_koncowy_ZAB_zadanie1A.ipynb
     ```
     *(Opcjonalnie: otwórz `raport_ZAB_zadanie1A.html` w przeglądarce i zapisz jako PDF).*

2. **Kod programu i wydruk wyników:**
   - W pliku `kod_zadanie1A.py` funkcja `uruchom_punkt4(df_raw)` jest już zaimplementowana i korzysta z wyników Punktu 3 (bez powtarzania imputacji). Uzupełnij jedynie `uruchom_punkt5(df_raw)` w miejscu oznaczonym `# TODO (ZESPÓŁ – PUNKT 5)`.
   - Uruchom skrypt (z katalogu `ZABMID/`), co automatycznie zaktualizuje `wydruk_wynikow_zadanie1A.txt`:
     ```powershell
     .\.venv\Scripts\python.exe kod_zadanie1A.py
     ```

3. **Paczka do oddania na MS Teams (3 obowiązkowe elementy):**
   - `raport_ZAB_zadanie1A.html` *(lub wyeksportowany z niego `.pdf`)*
   - `kod_zadanie1A.py` *(wraz z modułem `src/punkt3_ancova_mi.py` i ścieżką `CSV_FILE` na samej górze pliku)*
   - `wydruk_wynikow_zadanie1A.txt`
