# -*- coding: utf-8 -*-
"""
ZAAWANSOWANA ANALITYKA BIZNESOWA – METODY IMPUTACJI DANYCH
Zadanie nr 1, wersja A (zmienna ciągła)
Autorzy: Iwo Ceglarek, Maciek Staniec, Miłosz Granat, Piotr Szuta 121583, Szymon Józefczyk

Główny, w pełni odtwarzalny skrypt uruchamiający analizy dla całego zadania:
  - PUNKT 1: Analiza struktury danych oraz zakresu i rozkładu braków (wykresy fig1–fig5)
  - PUNKT 2: Ocena zależności występowania braków od cech (Welch t, chi2, regresja logistyczna)
  - PUNKT 3: Model ANCOVA (MCAR / MAR / MNAR 1D i 2D tipping-point; wykresy fig6–fig9)
  - PUNKT 4: [Miejsce przygotowane do uzupełnienia przez zespół]
  - PUNKT 5: [Miejsce przygotowane do uzupełnienia przez zespół]

Uruchomienie:
    .venv\\Scripts\\python.exe kod_zadanie1A.py

Wynik główny:
    wydruk_wynikow_zadanie1A.txt – pełny wydruk wyników z konsoli (załącznik na MS Teams),
    figures/fig1_* … fig9_*     – komplet wykresów (300 dpi),
    results/tab_p3_*.csv         – tabele szczegółowe dla Punktu 3.
"""
from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path

if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")  # zapis wykresów do plików bez otwierania okien GUI

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

# =============================================================================
# 0. KONFIGURACJA ŚCIEŻEK I PARAMETRÓW GLOBALNYCH
#    Aby odtworzyć wszystkie wyniki po przeniesieniu danych lub projektu,
#    wystarczy zmienić wyłącznie ścieżkę CSV_FILE (lub katalog bazowy ROOT).
# =============================================================================
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CSV_FILE = DATA / "ka01.csv"          # <-- ŚCIEŻKA DO ZBIORU DANYCH (zmień w razie potrzeby)
FIG = ROOT / "figures"
RES = ROOT / "results"

SEED = 42            # ziarno generatora losowego (pełna powtarzalność imputacji)
M_IMP = 100          # liczba wielokrotnych imputacji (m)
ALPHA = 0.05         # poziom istotności statystycznej
DPI = 300            # rozdzielczość zapisywanych wykresów

PAL = {"TRT=1 (czynnik)": "#2a6f97", "TRT=2 (kontrola)": "#c1121f"}

# Dołączenie modułu z implementacją Punktu 3 (znajdującego się w podkatalogu src/)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import punkt3_ancova_mi as p3

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


class _Tee(io.TextIOBase):
    """Jednoczesne wypisywanie wyników na konsolę oraz do pliku tekstowego."""

    def __init__(self, *strumienie):
        self.strumienie = strumienie

    def write(self, s: str) -> int:
        for st in self.strumienie:
            st.write(s)
        return len(s)

    def flush(self) -> None:
        for st in self.strumienie:
            st.flush()


def _naglowek(tekst: str) -> None:
    print("\n" + "=" * 100 + f"\n{tekst}\n" + "=" * 100)


def _opis_ciaglej(s: pd.Series) -> pd.Series:
    """Zestaw statystyk opisowych dla jednej cechy ciągłej (z pominięciem braków)."""
    s = s.dropna()
    return pd.Series({
        "n": int(s.size),
        "śr": s.mean(),
        "sd": s.std(ddof=1),
        "min": s.min(),
        "Q1": s.quantile(0.25),
        "med": s.median(),
        "Q3": s.quantile(0.75),
        "max": s.max(),
        "skośność": stats.skew(s),
        "kurtoza": stats.kurtosis(s),
    })


# =============================================================================
# PUNKT 1 – Analiza struktury danych oraz zakresu i rozkładu braków
# =============================================================================
def uruchom_punkt1(df_raw: pd.DataFrame) -> pd.DataFrame:
    _naglowek("PUNKT 1 – ANALIZA STRUKTURY DANYCH ORAZ ZAKRESU I ROZKŁADU BRAKÓW")
    df = df_raw.copy()
    df["chg"] = df["y_t2"] - df["y_t1"]
    df["trt_lab"] = df["trt"].map({1: "TRT=1 (czynnik)", 2: "TRT=2 (kontrola)"})

    ciagle = ["y_t1", "x1", "y_t2", "chg"]
    skokowe = ["trt", "x2"]
    kol_zrodlowe = ["y_t1", "x1", "trt", "x2", "y_t2"]

    print(f"Wymiary zbioru: {df_raw.shape[0]} obserwacji, {df_raw.shape[1]} zmiennych źródłowych")

    # 1.1 Struktura cech ciągłych (ogółem oraz wg grup TRT)
    print("\n1.1. Struktura cech ciągłych (ogółem):")
    struktura = df[ciagle].apply(_opis_ciaglej).T
    struktura["braki"] = df[ciagle].isna().sum().values
    print(struktura.round(3).to_string())

    print("\n1.2. Cechy ciągłe w podziale na grupy TRT:")
    for v in ciagle:
        print(f"\n-- {v} wg grup TRT --")
        print(df.groupby("trt_lab")[v].apply(_opis_ciaglej).unstack().round(3).to_string())

    # 1.3 Struktura cech skokowych
    print("\n1.3. Struktura cech skokowych:")
    for v in skokowe:
        t = df[v].value_counts().sort_index()
        print(f"\n{v}:")
        print(pd.DataFrame({"n": t, "%": (t / len(df) * 100).round(2)}).to_string())

    print("\nTabela krzyżowa trt × x2:")
    print(pd.crosstab(df["trt"], df["x2"], margins=True).to_string())

    # 1.4 Zakres i rozkład braków danych
    print("\n1.4. Zakres braków danych:")
    miss = df[kol_zrodlowe].isna().sum()
    print(pd.DataFrame({"n_braków": miss, "%": (miss / len(df) * 100).round(2)}).to_string())

    print("\nBraki per obserwacja (liczba brakujących komórek w wierszu):")
    print(df[kol_zrodlowe].isna().sum(axis=1).value_counts().sort_index().to_string())

    print("\nRozkład braków y_t2 wg cech skokowych:")
    for v in ["trt_lab", "x2"]:
        g = df.groupby(v)["y_t2"].apply(
            lambda s: pd.Series({"n": s.size, "braki": s.isna().sum(), "%_braków": round(s.isna().mean() * 100, 2)})
        )
        print(f"\nwg {v}:")
        print(g.unstack().to_string())

    # 1.5 Generowanie wykresów Punktu 1 (fig1–fig5, 300 dpi)
    FIG.mkdir(exist_ok=True)

    # fig1: macierz i kompletność braków
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    msno.matrix(df[kol_zrodlowe], ax=ax[0], sparkline=False, fontsize=12)
    ax[0].set_title("Macierz braków (wzorzec)")
    msno.bar(df[kol_zrodlowe], ax=ax[1], fontsize=12, color="#2a6f97")
    ax[1].set_title("Kompletność zmiennych")
    plt.tight_layout()
    fig.savefig(FIG / "fig1_braki_wzorzec.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # fig2: odsetek braków y_t2 wg grup
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for a, (var, lab) in zip(ax, [("trt_lab", "TRT"), ("x2", "X2")]):
        m = df.groupby(var)["y_t2"].apply(lambda s: s.isna().mean() * 100)
        x_labels = [f"X2={k}" for k in m.index] if var == "x2" else m.index.astype(str)
        a.bar(x_labels, m.values, color=["#2a6f97", "#c1121f"])
        a.axhline(df["y_t2"].isna().mean() * 100, ls="--", c="gray", label="ogółem 32,4%")
        a.set_ylim(0, 42)
        a.set_ylabel("% braków y_t2")
        a.set_title(f"Odsetek braków y_t2 wg {lab}")
        a.legend(loc="upper right")
        for i, val in enumerate(m.values):
            a.text(
                i, val + 0.8, f"{val:.1f}%", ha="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
            )
    plt.tight_layout()
    fig.savefig(FIG / "fig2_braki_grupy.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # fig3: rozkłady cech ciągłych wg grup
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for a, v in zip(ax, ["y_t1", "x1", "y_t2"]):
        for lab, g in df.groupby("trt_lab"):
            sns.histplot(
                g[v].dropna(), ax=a, kde=True, stat="density",
                color=PAL[lab], alpha=0.35, label=lab, element="step",
            )
        a.set_title(v)
        a.set_ylabel("Gęstość")
        a.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(FIG / "fig3_rozklady.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # fig4: wykresy pudełkowe wg grup
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for a, v in zip(ax, ["y_t1", "y_t2", "chg"]):
        sns.boxplot(data=df, x="trt_lab", y=v, hue="trt_lab", ax=a, palette=PAL, width=0.5, legend=False)
        sns.stripplot(data=df, x="trt_lab", y=v, ax=a, color="black", alpha=0.15, size=3)
        a.set_title(v)
        a.set_xlabel("")
        if v == "chg":
            a.axhline(0, ls="--", c="gray")
    plt.tight_layout()
    fig.savefig(FIG / "fig4_boxploty.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # fig5: zależność y_t2 ~ y_t1 wg grup
    fig, ax = plt.subplots(figsize=(9, 7))
    for lab, g in df.groupby("trt_lab"):
        sns.regplot(data=g, x="y_t1", y="y_t2", ax=ax, color=PAL[lab], label=lab, scatter_kws=dict(alpha=0.5, s=35))
    ax.plot([7, 15], [7, 15], ls=":", c="gray", label="y_t2 = y_t1 (brak zmiany)")
    ax.set_title("Zależność y_t2 ~ y_t1 wg grup")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG / "fig5_scatter.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print("\nZapisano wykresy Punktu 1: fig1_braki_wzorzec.png … fig5_scatter.png (300 dpi)")
    return df


# =============================================================================
# PUNKT 2 – Ocena zależności występowania braków danych od pozostałych cech
# =============================================================================
def uruchom_punkt2(df_raw: pd.DataFrame) -> None:
    _naglowek("PUNKT 2 – OCENA ZALEŻNOŚCI WYSTĘPOWANIA BRAKÓW DANYCH OD POZOSTAŁYCH CECH")
    df = df_raw.copy()
    df["R_y_t2"] = df["y_t2"].isna().astype(int)

    # 2.1 Porównanie średnich cech ciągłych (test t Welcha + korekta Holma)
    continuous_results = []
    for variable in ["y_t1", "x1"]:
        group_missing = df.loc[df["R_y_t2"] == 1, variable]
        group_observed = df.loc[df["R_y_t2"] == 0, variable]
        test = stats.ttest_ind(group_missing, group_observed, equal_var=False)
        difference = group_missing.mean() - group_observed.mean()
        se = np.sqrt(group_missing.var(ddof=1) / len(group_missing) + group_observed.var(ddof=1) / len(group_observed))
        df_welch = test.df
        critical_value = stats.t.ppf(0.975, df_welch)
        continuous_results.append({
            "Cecha": variable,
            "Średnia R=1": group_missing.mean(),
            "Średnia R=0": group_observed.mean(),
            "Różnica (R=1 − R=0)": difference,
            "CI 2.5%": difference - critical_value * se,
            "CI 97.5%": difference + critical_value * se,
            "t": test.statistic,
            "df": df_welch,
            "p-value": test.pvalue,
        })
    continuous_df = pd.DataFrame(continuous_results)
    continuous_df["p-value (Holm)"] = multipletests(continuous_df["p-value"], method="holm")[1]
    print("\n2.1. Porównanie średnich cech ciągłych w grupach R=1 i R=0 (test t Welcha):")
    print(continuous_df.round(4).to_string(index=False))

    # 2.2 Tabele kontyngencji dla cech skokowych (test chi-kwadrat + współczynnik phi + Holm)
    print("\n2.2. Zależność wskaźnika braków R_y_t2 od cech skokowych:")
    categorical_results = []
    for variable in ["trt", "x2"]:
        table = pd.crosstab(df[variable], df["R_y_t2"])
        chi2, p_value, dof, expected = stats.chi2_contingency(table, correction=False)
        phi = np.sqrt(chi2 / table.to_numpy().sum())
        print(f"\nTabela kontyngencji ({variable} × R_y_t2):")
        print(table.rename(columns={0: "R=0 (obs.)", 1: "R=1 (brak)"}).to_string())
        categorical_results.append({
            "Cecha": variable,
            "Chi2": chi2,
            "df": dof,
            "p-value": p_value,
            "phi": phi,
            "Min. licz. oczekiwana": expected.min(),
        })
    categorical_df = pd.DataFrame(categorical_results)
    categorical_df["p-value (Holm)"] = multipletests(categorical_df["p-value"], method="holm")[1]
    print("\nPodsumowanie testów niezależności chi-kwadrat:")
    print(categorical_df.round(4).to_string(index=False))

    # 2.3 Regresja logistyczna: R_y_t2 ~ y_t1 + x1 + C(trt) + C(x2) (błędy odporne HC3)
    model = smf.logit("R_y_t2 ~ y_t1 + x1 + C(trt) + C(x2)", data=df).fit(disp=False, cov_type="HC3")
    ci = model.conf_int()
    logit_results = pd.DataFrame({
        "Współczynnik": model.params,
        "OR": np.exp(model.params),
        "OR 95% CI dolna": np.exp(ci[0]),
        "OR 95% CI górna": np.exp(ci[1]),
        "p-value": model.pvalues,
    })
    print("\n2.3. Model regresji logistycznej dla wskaźnika braków R_y_t2 (HC3):")
    print(logit_results.round(4).to_string())

    null_model = smf.logit("R_y_t2 ~ 1", data=df).fit(disp=False)
    lr_stat = 2 * (model.llf - null_model.llf)
    lr_df = int(model.df_model - null_model.df_model)
    lr_p = stats.chi2.sf(lr_stat, lr_df)
    print(f"\nTest ilorazu wiarygodności (LR): stat = {lr_stat:.4f}, df = {lr_df}, "
          f"p-value = {lr_p:.4f}, Pseudo-R² McFaddena = {model.prsquared:.4f}")


# =============================================================================
# PUNKT 3 – Model ANCOVA (MCAR / MAR / MNAR 1D i 2D)
# =============================================================================
def uruchom_punkt3() -> dict:
    # Przekazanie bieżących ścieżek i parametrów z konfiguracji na górze skryptu
    p3.ROOT = ROOT
    p3.DATA = DATA
    p3.CSV_FILE = CSV_FILE
    p3.FIG = FIG
    p3.RES = RES
    p3.SEED = SEED
    p3.M_IMP = M_IMP
    p3.ALPHA = ALPHA
    p3.DPI = DPI
    return p3.uruchom_analize()


# =============================================================================
# PUNKT 4 – [MIEJSCE NA ROZWIĄZANIE PUNKTU 4 – DO UZUPEŁNIENIA PRZEZ ZESPÓŁ]
# =============================================================================
def uruchom_punkt4(df_raw: pd.DataFrame) -> None:
    _naglowek("PUNKT 4 – [MIEJSCE DO UZUPEŁNIENIA PRZEZ ZESPÓŁ]")
    # TODO (ZESPÓŁ – PUNKT 4):
    # Wklej tutaj kod realizujący wymagania Punktu 4 zadania
    # (korzystaj z parametrów df_raw, CSV_FILE, FIG, RES, SEED, ALPHA).
    print("[INFO] Sekcja Punktu 4 jest przygotowana do dopisania kodu przez pozostałych członków zespołu.")


# =============================================================================
# PUNKT 5 – [MIEJSCE NA ROZWIĄZANIE PUNKTU 5 – DO UZUPEŁNIENIA PRZEZ ZESPÓŁ]
# =============================================================================
def uruchom_punkt5(df_raw: pd.DataFrame) -> None:
    _naglowek("PUNKT 5 – [MIEJSCE DO UZUPEŁNIENIA PRZEZ ZESPÓŁ]")
    # TODO (ZESPÓŁ – PUNKT 5):
    # Wklej tutaj kod realizujący wymagania Punktu 5 zadania
    # oraz ewentualne końcowe zestawienia porównawcze.
    print("[INFO] Sekcja Punktu 5 jest przygotowana do dopisania kodu przez pozostałych członków zespołu.")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    plik_wy = ROOT / "wydruk_wynikow_zadanie1A.txt"
    stary_stdout = sys.stdout

    with open(plik_wy, "w", encoding="utf-8") as fh:
        sys.stdout = _Tee(stary_stdout, fh)
        try:
            with warnings.catch_warnings(record=True) as ostrz:
                warnings.simplefilter("always")
                warnings.filterwarnings(
                    "ignore",
                    message=r"vert: bool was deprecated.*",
                    module=r"seaborn\..*",
                )
                _naglowek("ZAAWANSOWANA ANALITYKA BIZNESOWA – ZADANIE NR 1 (WERSJA A)")
                print(f"Ścieżka danych: {CSV_FILE}")
                print(f"Parametry globalne: SEED = {SEED} | m = {M_IMP} | α = {ALPHA}")

                df_raw = pd.read_csv(CSV_FILE)
                uruchom_punkt1(df_raw)
                uruchom_punkt2(df_raw)
                uruchom_punkt3()
                uruchom_punkt4(df_raw)
                uruchom_punkt5(df_raw)

            _naglowek("PODSUMOWANIE OSTRZEŻEŃ I ZAPIS WYDRUKU")
            print(f"Liczba ostrzeżeń zarejestrowanych w trakcie całego skryptu: {len(ostrz)}")
            for w in ostrz:
                print(f"  {w.category.__name__}: {w.message} ({Path(w.filename).name}:{w.lineno})")
            print(f"Pełny wydruk wyników zapisano w: {plik_wy}")
        finally:
            sys.stdout = stary_stdout


if __name__ == "__main__":
    main()
