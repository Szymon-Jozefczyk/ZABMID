# -*- coding: utf-8 -*-
"""
ZAAWANSOWANA ANALITYKA BIZNESOWA – METODY IMPUTACJI DANYCH
Zadanie nr 1, wersja A (zmienna ciągła)
Autorzy: Iwo Ceglarek, Maciek Staniec, Miłosz Granat, Piotr Szuta 121583, Szymon Józefczyk

Główny, w pełni odtwarzalny skrypt uruchamiający analizy dla całego zadania:
  - PUNKT 1: Analiza struktury danych oraz zakresu i rozkładu braków (wykresy fig1–fig5)
  - PUNKT 2: Ocena zależności występowania braków od cech (Welch t, chi2, regresja logistyczna)
  - PUNKT 3: Model ANCOVA (MCAR / MAR / MNAR 1D i 2D tipping-point; wykresy fig6–fig9)
  - PUNKT 4: Porównanie i interpretacja wyników estymacji ANCOVA w kontekście braków
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
# PUNKT 4 – Podsumowanie wyników estymacji ANCOVA w kontekście braków danych
# =============================================================================
def uruchom_punkt4(df_raw: pd.DataFrame) -> None:
    """Podsumowanie wyników CC, MI/MAR i MNAR z punktu 3 (bez ponownej imputacji)."""
    _naglowek("PUNKT 4 – PODSUMOWANIE ESTYMACJI ANCOVA W KONTEKŚCIE BRAKÓW")

    por = pd.read_csv(RES / "tab_p3_porownanie.csv", index_col=0, encoding="utf-8-sig")
    mcar = pd.read_csv(RES / "tab_p3_ancova_mcar.csv", index_col=0, encoding="utf-8-sig")
    mar = pd.read_csv(RES / "tab_p3_ancova_mar.csv", index_col=0, encoding="utf-8-sig")
    skala = pd.read_csv(RES / "tab_p3_skala_delta.csv", index_col=0, encoding="utf-8-sig")
    kryt2d = pd.read_csv(RES / "tab_p3_tipping_2d_krytyczne.csv", encoding="utf-8-sig")

    tabela = pd.DataFrame({
        "MCAR (CC)": {
            "N": por.loc["N", "MCAR (CC)"],
            "β0": mcar.loc["β0 (wyraz wolny)", "Oszacowanie"],
            "β1 (TRT=1 vs 2)": por.loc["β1", "MCAR (CC)"],
            "SE(β1)": por.loc["SE(β1)", "MCAR (CC)"],
            "95% CI β1 — dolna": por.loc["CI 2.5%", "MCAR (CC)"],
            "95% CI β1 — górna": por.loc["CI 97.5%", "MCAR (CC)"],
            "p(β1)": por.loc["p-value", "MCAR (CC)"],
            "β2 (Y_T1)": por.loc["β2 (Y_T1)", "MCAR (CC)"],
            "p(β2)": mcar.loc["β2 (Y_T1)", "p-value"],
            "R²": por.loc["R²", "MCAR (CC)"],
            "FMI(β1)": np.nan,
        },
        "MAR (MI, FCS norm)": {
            "N": por.loc["N", "MAR (MI, FCS norm)"],
            "β0": mar.loc["β0 (wyraz wolny)", "Oszacowanie"],
            "β1 (TRT=1 vs 2)": por.loc["β1", "MAR (MI, FCS norm)"],
            "SE(β1)": por.loc["SE(β1)", "MAR (MI, FCS norm)"],
            "95% CI β1 — dolna": por.loc["CI 2.5%", "MAR (MI, FCS norm)"],
            "95% CI β1 — górna": por.loc["CI 97.5%", "MAR (MI, FCS norm)"],
            "p(β1)": por.loc["p-value", "MAR (MI, FCS norm)"],
            "β2 (Y_T1)": por.loc["β2 (Y_T1)", "MAR (MI, FCS norm)"],
            "p(β2)": mar.loc["β2 (Y_T1)", "p-value"],
            "R²": por.loc["R²", "MAR (MI, FCS norm)"],
            "FMI(β1)": por.loc["FMI(β1)", "MAR (MI, FCS norm)"],
        },
    })
    tabela.to_csv(RES / "tab_p4_podsumowanie.csv", encoding="utf-8-sig", index_label="Miara")

    def format_wartosc(wiersz: str, wartosc: float) -> str:
        if pd.isna(wartosc):
            return "—"
        if wiersz == "N":
            return f"{wartosc:.0f}"
        if wiersz.startswith("p("):
            return f"{wartosc:.2e}"
        if wiersz == "FMI(β1)":
            return f"{wartosc:.2%}"
        return f"{wartosc:.4f}"

    print("\n4.1. Tabela porównawcza estymacji ANCOVA:")
    do_wydruku = tabela.copy().astype(object)
    for w in tabela.index:
        for k in tabela.columns:
            do_wydruku.loc[w, k] = format_wartosc(w, tabela.loc[w, k])
    print(do_wydruku.to_string())

    # Także wartości p z punktu 2 są wyznaczane na aktualnych danych, nie wpisywane ręcznie.
    dane_logit = df_raw.copy()
    dane_logit["R_y_t2"] = dane_logit["y_t2"].isna().astype(int)
    logit = smf.logit("R_y_t2 ~ y_t1 + x1 + C(trt) + C(x2)", data=dane_logit).fit(
        disp=False, cov_type="HC3"
    )
    p_y1 = float(logit.pvalues["y_t1"])
    p_trt = float(logit.pvalues["C(trt)[T.2]"])
    n = len(df_raw)
    braki = int(df_raw["y_t2"].isna().sum())
    braki_t1 = int(df_raw.loc[df_raw["trt"] == 1, "y_t2"].isna().sum())
    braki_t2 = int(df_raw.loc[df_raw["trt"] == 2, "y_t2"].isna().sum())
    print(f"\n4.2. Braki końcowego pomiaru: {braki}/{n} ({100 * braki/n:.2f}%), "
          f"w TRT=1: {braki_t1}, w TRT=2: {braki_t2}.")
    print(f"W modelu logistycznym: p(Y_T1)={p_y1:.4f}, p(TRT)={p_trt:.4f}.")
    print("To argumenty przeciw MCAR, ale nie dowód MAR; na podstawie danych "
          "obserwowanych nie można wykluczyć MNAR.")

    b_cc = float(tabela.loc["β1 (TRT=1 vs 2)", "MCAR (CC)"])
    b_mi = float(tabela.loc["β1 (TRT=1 vs 2)", "MAR (MI, FCS norm)"])
    se_cc = float(tabela.loc["SE(β1)", "MCAR (CC)"])
    se_mi = float(tabela.loc["SE(β1)", "MAR (MI, FCS norm)"])
    print(f"\n4.3. Efekt TRT: β1(CC)={b_cc:.4f}; β1(MI)={b_mi:.4f}; "
          f"różnica MI-CC={b_mi - b_cc:+.4f}.")
    print(f"SE(CC)={se_cc:.4f}, SE(MI)={se_mi:.4f}; "
          f"zmniejszenie SE={(1 - se_mi/se_cc)*100:.2f}%.")
    print(f"β2(Y_T1) w MI={tabela.loc['β2 (Y_T1)', 'MAR (MI, FCS norm)']:.4f}, "
          f"p={tabela.loc['p(β2)', 'MAR (MI, FCS norm)']:.2e}.")
    print(f"FMI(β1) w MI={tabela.loc['FMI(β1)', 'MAR (MI, FCS norm)']:.2%}.")
    print("W obu wariantach 95% CI dla β1 nie obejmuje zera; odrzucamy H0: β1=0.")
    print(f"Alternatywna MI (sklearn): β1={por.loc['β1', 'MAR (MI, sklearn)']:.4f}.")

    delta_star = float(skala.loc["δ*", "wartość"])
    skala_sd = abs(float(skala.loc["δ* / SD(Y_T2 | TRT=1)", "wartość"]))
    print(f"\n4.4. MNAR 1D: δ*={delta_star:.4f} przy przesunięciu {braki_t1} "
          f"imputowanych Y_T2 w TRT=1 (|δ*|/SD={skala_sd:.2f}); na progu p≈0,05.")
    print("MNAR 2D: punkty graniczne zależne od przesunięcia δ2 w grupie kontrolnej:")
    print(kryt2d.round(4).to_string(index=False))
    print("Wniosek: dodatni efekt TRT jest stabilny w CC i MI/MAR, lecz istotność "
          "zależy od rozpatrywanego scenariusza MNAR.")
    print(f"Tabela punktu 4 zapisana w: {RES / 'tab_p4_podsumowanie.csv'}")

# =============================================================================
# PUNKT 5 – [MIEJSCE NA ROZWIĄZANIE PUNKTU 5 – DO UZUPEŁNIENIA PRZEZ ZESPÓŁ]
# =============================================================================

def uruchom_punkt5(df_raw: pd.DataFrame) -> None:
    """Diagnostyka wielokrotnej imputacji (FMI, RIV, RE z reguł Rubina oraz D_KS)."""
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge

    _naglowek("PUNKT 5 – MIARA DIAGNOSTYCZNA OCENY IMPUTACJI W KONTEKŚCIE MODELU ANCOVA")

    features = ["y_t1", "x1", "trt1", "x2", "y_t2"]
    df = df_raw[["y_t1", "x1", "trt", "x2", "y_t2"]].copy()
    df["trt1"] = (df["trt"] == 1).astype(int)
    obs_mask = df["y_t2"].notna()

    obs_y2 = df.loc[obs_mask, "y_t2"].to_numpy()
    obs_chg = (df.loc[obs_mask, "y_t2"] - df.loc[obs_mask, "y_t1"]).to_numpy()

    param_names = ["Intercept", "trt1", "y_t1"]
    labels = ["β0 (wyraz wolny)", "β1 (TRT=1 vs 2)", "β2 (Y_T1)"]

    q_list, u_list = [], []
    ks_y2_list, ks_chg_list, wass_y2_list, wass_chg_list = [], [], [], []
    imp_chg_samples = []

    for m in range(M_IMP):
        imp = IterativeImputer(
            estimator=BayesianRidge(),
            sample_posterior=True,
            random_state=SEED + m,
            max_iter=20,
        )
        df_imp = pd.DataFrame(imp.fit_transform(df[features]), columns=features)
        df_imp["trt1"] = (df_imp["trt1"] >= 0.5).astype(int)
        df_imp["chg"] = df_imp["y_t2"] - df_imp["y_t1"]

        mod = smf.ols("chg ~ trt1 + y_t1", data=df_imp).fit()
        q_list.append(mod.params[param_names].to_numpy())
        u_list.append((mod.bse[param_names].to_numpy()) ** 2)

        imp_y2 = df_imp.loc[~obs_mask, "y_t2"].to_numpy()
        imp_chg = df_imp.loc[~obs_mask, "chg"].to_numpy()
        ks_y2_list.append(stats.ks_2samp(obs_y2, imp_y2).statistic)
        ks_chg_list.append(stats.ks_2samp(obs_chg, imp_chg).statistic)
        wass_y2_list.append(stats.wasserstein_distance(obs_y2, imp_y2))
        wass_chg_list.append(stats.wasserstein_distance(obs_chg, imp_chg))
        if m < 15:
            imp_chg_samples.append(imp_chg)

    q_arr = np.array(q_list)
    u_arr = np.array(u_list)
    n_obs = len(df)
    nu_com = n_obs - len(param_names)

    rows = []
    for idx, label in enumerate(labels):
        q_bar = float(np.mean(q_arr[:, idx]))
        u_bar = float(np.mean(u_arr[:, idx]))
        b_var = float(np.var(q_arr[:, idx], ddof=1))
        t_var = u_bar + (1.0 + 1.0 / M_IMP) * b_var
        se_tot = float(np.sqrt(t_var))
        riv = ((1.0 + 1.0 / M_IMP) * b_var) / u_bar

        nu_old = (M_IMP - 1) * (1.0 + 1.0 / riv) ** 2
        gamma_m = ((1.0 + 1.0 / M_IMP) * b_var) / t_var
        nu_obs = ((nu_com + 1.0) / (nu_com + 3.0)) * nu_com * (1.0 - gamma_m)
        nu_adj = (nu_old * nu_obs) / (nu_old + nu_obs)

        fmi = (riv + 2.0 / (nu_adj + 3.0)) / (riv + 1.0)
        re = 1.0 / (1.0 + fmi / M_IMP)
        p_val = float(2.0 * stats.t.sf(abs(q_bar / se_tot), df=nu_adj))

        rows.append({
            "Parametr": label,
            "Oszacowanie (Q̄)": q_bar,
            "SE całkowite (√T)": se_tot,
            "Ū (wewn.)": u_bar,
            "B (między.)": b_var,
            "RIV (r)": riv,
            "FMI (λ)": fmi,
            "RE": re,
            "p-value": p_val,
        })

    diag_df = pd.DataFrame(rows)
    diag_df.to_csv(RES / "tab_p5_diagnostyka.csv", index=False, encoding="utf-8-sig")

    print(f"\n5.1. Diagnostyka parametrów modelu ANCOVA wg reguł Rubina (m = {M_IMP}, SEED = {SEED}):")
    print(diag_df.round(4).to_string(index=False))

    print("\n5.2. Zgodność rozkładów wartości imputowanych vs obserwowanych:")
    print(f"  Średnia statystyka Kołmogorowa-Smirnowa D_KS (y_t2): {np.mean(ks_y2_list):.4f}")
    print(f"  Średnia statystyka Kołmogorowa-Smirnowa D_KS (chg):  {np.mean(ks_chg_list):.4f}")
    print(f"  Średni dystans Wassersteina W1 (y_t2):               {np.mean(wass_y2_list):.4f}")
    print(f"  Średni dystans Wassersteina W1 (chg):                {np.mean(wass_chg_list):.4f}")

    por_path = RES / "tab_p3_porownanie.csv"
    if por_path.exists():
        por = pd.read_csv(por_path, index_col=0, encoding="utf-8-sig")
        if "FMI(β1)" in por.index and "MAR (MI, FCS norm)" in por.columns:
            fmi_fcs = float(por.loc["FMI(β1)", "MAR (MI, FCS norm)"])
            fmi_br = float(diag_df.loc[diag_df["Parametr"] == "β1 (TRT=1 vs 2)", "FMI (λ)"].iloc[0])
            print(f"\n5.3. Porównanie FMI(β1): FCS norm (Punkt 3) = {fmi_fcs:.2%} | BayesianRidge MI = {fmi_br:.2%}")

    # Wykres diagnostyczny fig10 (300 dpi)
    FIG.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))

    sns.kdeplot(obs_chg, ax=ax[0], color="#2a6f97", lw=2.5, label=f"Obserwowane CHG (n={obs_mask.sum()})")
    for i, s_chg in enumerate(imp_chg_samples):
        sns.kdeplot(
            s_chg, ax=ax[0], color="#c1121f", alpha=0.22, lw=1.0,
            label=f"Imputowane CHG (pierwsze 15 z m={M_IMP})" if i == 0 else None,
        )
    ax[0].axvline(0, ls=":", c="gray")
    ax[0].set_title(f"Zgodność rozkładów przyrostu CHG (śr. D_KS = {np.mean(ks_chg_list):.4f})")
    ax[0].set_xlabel("CHG = Y_T2 − Y_T1")
    ax[0].set_ylabel("Gęstość")
    ax[0].legend(fontsize=9)

    q1_bar = float(diag_df.loc[1, "Oszacowanie (Q̄)"])
    fmi_b1 = float(diag_df.loc[1, "FMI (λ)"])
    sns.histplot(q_arr[:, 1], kde=True, ax=ax[1], color="#2a6f97", alpha=0.45, bins=18)
    ax[1].axvline(q1_bar, color="#c1121f", lw=2, ls="--", label=f"Q̄(β1) = {q1_bar:.4f}")
    ax[1].set_title(f"Zmienność międzyimputacyjna β1 (m={M_IMP}, FMI = {fmi_b1:.2%})")
    ax[1].set_xlabel("Oszacowanie β1 (TRT=1 vs TRT=2)")
    ax[1].set_ylabel("Liczba imputacji")
    ax[1].legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(FIG / "fig10_diagnostyka_imputacji.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"\nZapisano tabelę: {RES / 'tab_p5_diagnostyka.csv'}")
    print(f"Zapisano wykres: {FIG / 'fig10_diagnostyka_imputacji.png'} ({DPI} dpi)")
    


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
