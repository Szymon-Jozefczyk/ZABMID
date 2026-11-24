# -*- coding: utf-8 -*-
"""
PUNKT 3 – Model analizy kowariancji (ANCOVA) przy założeniach MCAR i MAR
oraz korekta oszacowań w scenariuszu MNAR (δ-adjustment, tipping-point analysis).

Model analityczny (ANCOVA):
    CHG_i = β0 + β1·I(TRT_i = 1) + β2·Y_T1_i + ε_i,   CHG = Y_T2 − Y_T1,
    TRT = 2 (grupa kontrolna) jest kategorią referencyjną, więc β1 = różnica
    (TRT=1 − TRT=2) skorygowana o Y_T1.

Scenariusze:
    MCAR – analiza pełnych przypadków (complete-case, CC),
    MAR  – imputacja wielokrotna (FCS / MICE, bayesowska regresja liniowa
           „norm” z losowaniem parametrów z rozkładu a posteriori, osobno
           w warstwach TRT) + własna implementacja reguł Rubina,
    MNAR – przesunięcie δ wartości imputowanych Y_T2 (δ-adjustment)
           i wyznaczenie punktu krytycznego δ* (1D i 2D tipping point).

Uruchomienie (z dowolnego katalogu):
    .venv\\Scripts\\python.exe src\\punkt3_ancova_mi.py
Wyniki:
    results/punkt3_wydruk_konsoli.txt – pełny wydruk konsoli (załącznik),
    results/tab_p3_*.csv               – tabele wynikowe,
    figures/fig6_* … fig9_*            – wykresy (300 dpi).

Moduł można też importować w notebooku (notebooks/03_ancova_imputacja_mnar.ipynb).
"""
from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

if __name__ == "__main__":  # w trybie skryptu – zapis wykresów bez okien
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.colors import LogNorm, TwoSlopeNorm

# =============================================================================
# 0. WSPÓLNA DEFINICJA ŚCIEŻEK I STAŁYCH (jak w notebookach 01 i 02)
#    Po przeniesieniu projektu wystarczy zmienić wyłącznie ROOT.
# =============================================================================
ROOT = Path(__file__).resolve().parent.parent          # katalog ZABMID
DATA = ROOT / "data"
CSV_FILE = DATA / "ka01.csv"
FIG = ROOT / "figures"
RES = ROOT / "results"

SEED = 42            # ziarno bazowe; imputacja i używa ziarna SEED + i
M_IMP = 100          # liczba imputacji (m)
ALPHA = 0.05         # poziom istotności
KROK_DELTA = 0.5     # krok zgrubnej siatki δ w analizie tipping-point
DPI = 300            # rozdzielczość wykresów

# zmienne modelu imputacyjnego (imputacja odbywa się osobno w warstwach TRT)
ZM_IMPUTACJA = ["y_t1", "x1", "x2", "y_t2"]
ZM_BINARNE = ["x2"]      # zmienne 0/1 – w razie braków: bayesowska regresja logistyczna
WARSTWA = "trt"

# model ANCOVA – TRT=2 (kontrola) jako kategoria referencyjna
FORMULA_ANCOVA = "chg ~ C(trt, Treatment(reference=2)) + y_t1"
PARAM = ["β0 (wyraz wolny)", "β1 (TRT=1 vs TRT=2)", "β2 (Y_T1)"]

PAL = {"TRT=1 (czynnik)": "#2a6f97", "TRT=2 (kontrola)": "#c1121f"}
KOL_TRT = {1: PAL["TRT=1 (czynnik)"], 2: PAL["TRT=2 (kontrola)"]}

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


# =============================================================================
# 1. WCZYTANIE I PRZYGOTOWANIE DANYCH
# =============================================================================
def wczytaj_dane(csv_file: Path = CSV_FILE) -> pd.DataFrame:
    """Wczytuje zbiór grupy i tworzy zmienne pochodne.

    brak_y_t2 – maska logiczna ORYGINALNYCH braków Y_T2 (zachowana do MNAR),
    trt_ind   – 1 dla TRT=1 (czynnik), 0 dla TRT=2 (kontrola – referencja),
    chg       – Y_T2 − Y_T1 (NaN tam, gdzie brak Y_T2).
    """
    df = pd.read_csv(csv_file)
    df["brak_y_t2"] = df["y_t2"].isna()
    df["trt_ind"] = (df["trt"] == 1).astype(int)
    df["chg"] = df["y_t2"] - df["y_t1"]
    df["trt_lab"] = df["trt"].map({1: "TRT=1 (czynnik)", 2: "TRT=2 (kontrola)"})
    return df


def opis_brakow(df: pd.DataFrame) -> pd.DataFrame:
    """Liczba i odsetek braków w zmiennych źródłowych."""
    kol = ["y_t1", "x1", "trt", "x2", "y_t2"]
    miss = df[kol].isna().sum()
    return pd.DataFrame({"n_braków": miss, "%": (miss / len(df) * 100).round(2)})


# =============================================================================
# 2. REGUŁY RUBINA (własna implementacja)
# =============================================================================
def rubin(Q, U, nu_com, alpha: float = ALPHA, names=None) -> pd.DataFrame:
    """Łączenie wyników m imputacji regułami Rubina (1987).

    Q – oszacowania, macierz (m, k);  U – kwadraty błędów standardowych (m, k);
    nu_com – stopnie swobody analizy na danych kompletnych (skalar lub wektor k).

    Q̄ = (1/m) Σ Q_i,  W = Ū = (1/m) Σ U_i,  B = 1/(m−1) Σ (Q_i − Q̄)²,
    T = W + (1 + 1/m)·B,  RIV r = (1 + 1/m)·B / W,  λ = (1 + 1/m)·B / T,
    ν_old = (m − 1)/λ²,  ν_obs = (ν_com + 1)/(ν_com + 3)·ν_com·(1 − λ),
    ν_adj = ν_old·ν_obs / (ν_old + ν_obs)   (Barnard i Rubin, 1999),
    FMI = (r + 2/(ν_adj + 3)) / (1 + r).
    """
    Q = np.asarray(Q, dtype=float)
    U = np.asarray(U, dtype=float)
    if Q.ndim == 1:
        Q, U = Q[:, None], U[:, None]
    m = Q.shape[0]
    nu_com = np.broadcast_to(np.asarray(nu_com, dtype=float), Q.shape[1:])

    qbar = Q.mean(axis=0)
    w = U.mean(axis=0)
    b = Q.var(axis=0, ddof=1) if m > 1 else np.zeros_like(qbar)
    t_var = w + (1 + 1 / m) * b
    riv = (1 + 1 / m) * b / w
    lam = (1 + 1 / m) * b / t_var
    with np.errstate(divide="ignore", invalid="ignore"):
        nu_old = np.where(lam > 0, (m - 1) / lam**2, np.inf)
        nu_obs = (nu_com + 1) / (nu_com + 3) * nu_com * (1 - lam)
        nu_adj = np.where(np.isinf(nu_old), nu_obs, nu_old * nu_obs / (nu_old + nu_obs))
    fmi = (riv + 2 / (nu_adj + 3)) / (1 + riv)

    se = np.sqrt(t_var)
    t_stat = qbar / se
    p = 2 * stats.t.sf(np.abs(t_stat), nu_adj)
    q = stats.t.ppf(1 - alpha / 2, nu_adj)
    return pd.DataFrame(
        {
            "Oszacowanie": qbar, "SE": se, "t": t_stat, "df (ν_adj)": nu_adj,
            "p-value": p, "CI 2.5%": qbar - q * se, "CI 97.5%": qbar + q * se,
            "W": w, "B": b, "T": t_var, "RIV": riv, "λ": lam, "FMI": fmi,
        },
        index=names,
    )


# =============================================================================
# 3. MODEL ANCOVA – dopasowanie, LS-means, średnie surowe
# =============================================================================
def dopasuj_ancova(d: pd.DataFrame):
    """OLS: chg ~ C(trt, Treatment(reference=2)) + y_t1 (statsmodels)."""
    res = smf.ols(FORMULA_ANCOVA, data=d).fit()
    nazwy = list(res.params.index)
    # kontrola kolejności parametrów: [Intercept, trt(T.1), y_t1]
    assert nazwy[0] == "Intercept" and "trt" in nazwy[1] and "T.1" in nazwy[1] \
        and nazwy[2] == "y_t1", nazwy
    return res


def lsmeans(res, y_t1_ref: float):
    """LS-means CHG dla TRT=1 i TRT=2 przy Y_T1 = y_t1_ref: L·β, Var = L·V·L'."""
    L = np.array([[1.0, 1.0, y_t1_ref], [1.0, 0.0, y_t1_ref]])
    est = L @ res.params.to_numpy()
    var = np.einsum("ij,jk,ik->i", L, res.cov_params().to_numpy(), L)
    return est, var


def srednie_surowe(d: pd.DataFrame):
    """Surowe średnie CHG w grupach TRT=1 i TRT=2 oraz ich wariancje (s²/n)."""
    g = d.groupby("trt")["chg"]
    mean = g.mean().loc[[1, 2]].to_numpy()
    var = (g.var(ddof=1) / g.count()).loc[[1, 2]].to_numpy()
    n = g.count().loc[[1, 2]].to_numpy()
    return mean, var, n


WIERSZE_SR = ["CHG surowa – TRT=1", "CHG surowa – TRT=2",
              "LS-mean CHG – TRT=1", "LS-mean CHG – TRT=2"]


# =============================================================================
# 4. SCENARIUSZ MCAR – analiza pełnych przypadków
# =============================================================================
def analiza_mcar(df: pd.DataFrame, alpha: float = ALPHA) -> dict:
    cc = df.loc[~df["brak_y_t2"]].copy()
    res = dopasuj_ancova(cc)
    ci = res.conf_int(alpha=alpha).to_numpy()
    wsp = pd.DataFrame(
        {
            "Oszacowanie": res.params.to_numpy(), "SE": res.bse.to_numpy(),
            "t": res.tvalues.to_numpy(), "df": res.df_resid,
            "p-value": res.pvalues.to_numpy(), "CI 2.5%": ci[:, 0], "CI 97.5%": ci[:, 1],
        },
        index=PARAM,
    )
    y1_ref = cc["y_t1"].mean()          # średnia bazowa w podpróbie kompletnej (CC)
    ls_est, ls_var = lsmeans(res, y1_ref)
    sr_est, sr_var, n_g = srednie_surowe(cc)
    est = np.r_[sr_est, ls_est]
    se = np.sqrt(np.r_[sr_var, ls_var])
    dfs = np.r_[n_g - 1, [res.df_resid] * 2]
    q = stats.t.ppf(1 - alpha / 2, dfs)
    srednie = pd.DataFrame(
        {"Oszacowanie": est, "SE": se, "CI 2.5%": est - q * se, "CI 97.5%": est + q * se,
         "n": np.r_[n_g, [len(cc)] * 2]},
        index=WIERSZE_SR,
    )
    return {"model": res, "wsp": wsp, "srednie": srednie, "n": int(res.nobs),
            "r2": res.rsquared, "y1_ref": y1_ref}


# =============================================================================
# 5. SCENARIUSZ MAR – imputacja wielokrotna FCS (MICE)
# =============================================================================
def _macierz_X(dat: pd.DataFrame, predyktory: list[str]) -> np.ndarray:
    return np.column_stack([np.ones(len(dat)), dat[predyktory].to_numpy(float)])


def losuj_norm(X_obs, y_obs, X_mis, rng: np.random.Generator) -> np.ndarray:
    """Bayesowska regresja liniowa (Rubin 1987, alg. „norm” w pakiecie mice).

    1) β̂ = (X'X)⁻¹X'y,  SSR = ||y − Xβ̂||²,
    2) σ*² = SSR / χ²_(n−p)                     – losowanie wariancji resztowej,
    3) β* = β̂ + σ*·chol((X'X)⁻¹)·z,  z ~ N(0, I) – losowanie współczynników,
    4) y_imp = X_mis·β* + σ*·ε,  ε ~ N(0, 1)    – losowy błąd resztowy.
    Kroki 2–3 odzwierciedlają niepewność parametrów, krok 4 – zmienność losową.
    """
    n, p = X_obs.shape
    xtx_inv = np.linalg.inv(X_obs.T @ X_obs)
    beta_hat = xtx_inv @ X_obs.T @ y_obs
    ssr = float(np.sum((y_obs - X_obs @ beta_hat) ** 2))
    sigma2 = ssr / rng.chisquare(n - p)
    chol = np.linalg.cholesky(xtx_inv)
    beta = beta_hat + np.sqrt(sigma2) * chol @ rng.standard_normal(p)
    return X_mis @ beta + np.sqrt(sigma2) * rng.standard_normal(X_mis.shape[0])


def losuj_logit(X_obs, y_obs, X_mis, rng: np.random.Generator) -> np.ndarray:
    """Bayesowska regresja logistyczna (alg. „logreg”) dla zmiennych 0/1:
    β* ~ N(β̂_ML, V̂(β̂)) (aproksymacja normalna rozkładu a posteriori),
    y_imp ~ Bernoulli(logit⁻¹(X_mis·β*))."""
    fit = sm.GLM(y_obs, X_obs, family=sm.families.Binomial()).fit()
    chol = np.linalg.cholesky(np.asarray(fit.cov_params()))
    beta = np.asarray(fit.params) + chol @ rng.standard_normal(X_obs.shape[1])
    prob = 1 / (1 + np.exp(-(X_mis @ beta)))
    return rng.binomial(1, prob).astype(float)


def imputuj_fcs(df: pd.DataFrame, rng: np.random.Generator, n_iter: int = 10) -> pd.DataFrame:
    """Jedna kompletna imputacja metodą FCS (MICE), osobno w warstwach TRT.

    W każdej warstwie TRT ∈ {1, 2} każda niekompletna zmienna jest imputowana
    z modelu warunkowego względem wszystkich pozostałych zmiennych z ZM_IMPUTACJA
    (ciągłe – „norm”, binarne – „logreg”). Imputacja w warstwach jest
    równoważna pełnym interakcjom wszystkich predyktorów z TRT i pozwala na
    odrębne wariancje resztowe w grupach. Przy jednej niekompletnej zmiennej
    (wzorzec monotoniczny – nasz przypadek: braki tylko w Y_T2) algorytm jest
    zbieżny po jednej iteracji i jest dokładnym losowaniem z rozkładu
    predykcyjnego a posteriori.
    """
    wynik = df[ZM_IMPUTACJA].astype(float).copy()
    for _, idx in df.groupby(WARSTWA).groups.items():
        dat = wynik.loc[idx].copy()
        brak = dat.isna()
        niepelne = [v for v in ZM_IMPUTACJA if brak[v].any()]
        if len(niepelne) > 1:            # start: losowanie z wartości zaobserwowanych
            for v in niepelne:
                obs_v = dat.loc[~brak[v], v].to_numpy()
                dat.loc[brak[v], v] = rng.choice(obs_v, int(brak[v].sum()))
        for _ in range(1 if len(niepelne) <= 1 else n_iter):
            for v in niepelne:
                pred = [c for c in ZM_IMPUTACJA if c != v]
                X = _macierz_X(dat, pred)
                obs = ~brak[v].to_numpy()
                y = dat[v].to_numpy(float)
                losuj = losuj_logit if v in ZM_BINARNE else losuj_norm
                dat.loc[brak[v], v] = losuj(X[obs], y[obs], X[~obs], rng)
        wynik.loc[idx] = dat
    return wynik


def wielokrotna_imputacja(df: pd.DataFrame, m: int = M_IMP, seed: int = SEED) -> np.ndarray:
    """m niezależnych imputacji (ziarna SEED + i). Zwraca macierz Y_T2 (n, m)."""
    Y2 = np.empty((len(df), m))
    for i in range(m):
        rng = np.random.default_rng(seed + i)
        Y2[:, i] = imputuj_fcs(df, rng)["y_t2"].to_numpy()
    return Y2


def wielokrotna_imputacja_sklearn(df: pd.DataFrame, m: int = M_IMP, seed: int = SEED) -> np.ndarray:
    """Weryfikacja krzyżowa: IterativeImputer(BayesianRidge, sample_posterior=True),
    osobno w warstwach TRT, m imputacji z ziarnami SEED + i."""
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge

    Y2 = np.empty((len(df), m))
    for i in range(m):
        for _, idx in df.groupby(WARSTWA).indices.items():
            imp = IterativeImputer(estimator=BayesianRidge(), sample_posterior=True,
                                   max_iter=10, random_state=seed + i)
            arr = imp.fit_transform(df.iloc[idx][ZM_IMPUTACJA])
            Y2[idx, i] = arr[:, ZM_IMPUTACJA.index("y_t2")]
    return Y2


def zbior_imputowany(df: pd.DataFrame, Y2: np.ndarray, i: int) -> pd.DataFrame:
    """i-ty kompletny zbiór; CHG liczone PO imputacji (passive imputation)."""
    d = df[["y_t1", "x1", "x2", "trt", "trt_ind", "brak_y_t2"]].copy()
    d["y_t2"] = Y2[:, i]
    d["chg"] = d["y_t2"] - d["y_t1"]
    return d


def analiza_mar(df: pd.DataFrame, Y2: np.ndarray, alpha: float = ALPHA) -> dict:
    """ANCOVA na każdym z m zbiorów + łączenie regułami Rubina."""
    m, n = Y2.shape[1], len(df)
    y1_ref = df["y_t1"].mean()
    Qb, Ub = np.empty((m, 3)), np.empty((m, 3))
    Qs, Us = np.empty((m, 4)), np.empty((m, 4))
    r2, s2 = np.empty(m), np.empty(m)
    for i in range(m):
        d = zbior_imputowany(df, Y2, i)
        res = dopasuj_ancova(d)
        Qb[i], Ub[i] = res.params.to_numpy(), res.bse.to_numpy() ** 2
        ls_est, ls_var = lsmeans(res, y1_ref)
        sr_est, sr_var, n_g = srednie_surowe(d)
        Qs[i], Us[i] = np.r_[sr_est, ls_est], np.r_[sr_var, ls_var]
        r2[i], s2[i] = res.rsquared, res.mse_resid
    wsp = rubin(Qb, Ub, nu_com=n - 3, alpha=alpha, names=PARAM)
    srednie = rubin(Qs, Us, nu_com=np.r_[n_g - 1, [n - 3] * 2], alpha=alpha, names=WIERSZE_SR)
    # R² łączone przez transformację Fishera (Harel, 2009)
    r2_pool = np.tanh(np.mean(np.arctanh(np.sqrt(r2)))) ** 2
    return {"wsp": wsp, "srednie": srednie, "Q_beta": Qb, "U_beta": Ub, "r2": r2_pool,
            "rmse": float(np.sqrt(s2.mean())), "n": n, "m": m, "y1_ref": y1_ref}


# =============================================================================
# 6. SZYBKA ANCOVA (algebra macierzowa) – do siatek δ w MNAR
# =============================================================================
class SzybkaANCOVA:
    """Wektorowe OLS dla wszystkich m imputacji naraz.
    Macierz X = [1, I(TRT=1), Y_T1] jest kompletna i wspólna dla wszystkich
    imputacji i przesunięć δ, więc (X'X)⁻¹X' liczymy tylko raz."""

    def __init__(self, df: pd.DataFrame):
        self.y_t1 = df["y_t1"].to_numpy(float)
        self.X = np.column_stack([np.ones(len(df)), df["trt_ind"].to_numpy(float), self.y_t1])
        self.n, self.p = self.X.shape
        self.xtx_inv = np.linalg.inv(self.X.T @ self.X)
        self.H = self.xtx_inv @ self.X.T

    def dopasuj(self, Y2: np.ndarray):
        """Y2 (n, m) → (B, V): oszacowania i wariancje parametrów, macierze (m, p)."""
        chg = Y2 - self.y_t1[:, None]
        B = self.H @ chg
        resid = chg - self.X @ B
        s2 = (resid**2).sum(axis=0) / (self.n - self.p)
        V = s2[None, :] * np.diag(self.xtx_inv)[:, None]
        return B.T, V.T


# =============================================================================
# 7. SCENARIUSZ MNAR – δ-adjustment i tipping-point analysis
# =============================================================================
def przesun_mnar(Y2, brak, trt, delta1: float, delta2: float = 0.0) -> np.ndarray:
    """Y_T2_MNAR = Y_T2_MAR + δ – WYŁĄCZNIE dla pierwotnie brakujących Y_T2:
    δ1 w grupie TRT=1, δ2 w grupie TRT=2. Wartości zaobserwowane bez zmian."""
    Y = Y2.copy()
    Y[brak & (trt == 1), :] += delta1
    Y[brak & (trt == 2), :] += delta2
    return Y


def beta1_mnar(model: SzybkaANCOVA, Y2, brak, trt, delta1, delta2=0.0,
               alpha: float = ALPHA) -> pd.Series:
    """β1 połączone regułami Rubina dla zadanej pary (δ1, δ2)."""
    B, V = model.dopasuj(przesun_mnar(Y2, brak, trt, delta1, delta2))
    return rubin(B[:, [1]], V[:, [1]], nu_com=model.n - model.p, alpha=alpha).iloc[0]


def _nz(v: float) -> float:
    return 0.0 if abs(v) < 1e-9 else float(v)


def _wiersz(delta, r: pd.Series, alpha=ALPHA) -> dict:
    return {"δ": _nz(delta), "β1": _nz(r["Oszacowanie"]), "SE": float(r["SE"]),
            "CI 2.5%": _nz(r["CI 2.5%"]), "CI 97.5%": _nz(r["CI 97.5%"]),
            "p-value": float(r["p-value"]), "FMI": float(r["FMI"]),
            "Decyzja (α=0.05)": "odrzucamy H0" if r["p-value"] < alpha else "brak podstaw do odrzucenia H0"}


def tipping_point_1d(model: SzybkaANCOVA, Y2, brak, trt, wsp_mar: pd.DataFrame,
                     krok: float = KROK_DELTA, alpha: float = ALPHA, max_krokow: int = 400) -> dict:
    """Jednowymiarowa analiza punktu krytycznego (przesuwamy braki w TRT=1).

    1) kierunek: jeśli β1 (MAR) istotne – w stronę utraty istotności (−sign β1),
       jeśli nieistotne – w stronę uzyskania istotności (+sign β1);
    2) zgrubna siatka δ = 0, ±krok, ±2·krok, … rozszerzana aż do zmiany decyzji;
    3) zagęszczenie siatki (krok/10) w przedziale przejścia;
    4) dokładny δ*: pierwiastek równania p(δ) = α (metoda Brenta) – p(δ) jest
       ciągłą funkcją δ, bo zbiory imputowane są stałe.
    """
    b1, p0 = wsp_mar.iloc[1]["Oszacowanie"], wsp_mar.iloc[1]["p-value"]
    istotny_mar = p0 < alpha
    kierunek = -np.sign(b1) if istotny_mar else np.sign(b1)

    zgrubna, k = [], 0
    while True:
        d = 0.0 if k == 0 else float(kierunek * krok * k)
        r = beta1_mnar(model, Y2, brak, trt, d, alpha=alpha)
        zgrubna.append(_wiersz(d, r, alpha))
        if (r["p-value"] < alpha) != istotny_mar:
            break
        k += 1
        if k > max_krokow:
            raise RuntimeError("Siatka δ nie przecięła progu α – zwiększ max_krokow lub krok.")
    for j in (1, 2):                                   # dwa punkty za progiem (do tabeli)
        d = float(kierunek * krok * (k + j))
        zgrubna.append(_wiersz(d, beta1_mnar(model, Y2, brak, trt, d, alpha=alpha), alpha))
    zgrubna = pd.DataFrame(zgrubna)

    lo, hi = sorted([kierunek * krok * (k - 1), kierunek * krok * k])
    gesta = pd.DataFrame([_wiersz(float(d), beta1_mnar(model, Y2, brak, trt, float(d), alpha=alpha), alpha)
                          for d in np.round(np.arange(lo, hi + 1e-9, krok / 10), 10)])
    f = lambda d: beta1_mnar(model, Y2, brak, trt, d, alpha=alpha)["p-value"] - alpha  # noqa: E731
    delta_star = optimize.brentq(f, lo, hi, xtol=1e-10, rtol=1e-12)
    r_star = beta1_mnar(model, Y2, brak, trt, delta_star, alpha=alpha)

    # β1 jest liniowe w δ: β1(δ) = β1_MAR + c·δ, c = [(X'X)⁻¹X']_β1 · 1(brak ∧ TRT=1)
    c = float(model.H[1] @ (brak & (trt == 1)).astype(float))
    delta_zero = -b1 / c
    return {"kierunek": kierunek, "istotny_mar": istotny_mar, "zgrubna": zgrubna, "gesta": gesta,
            "delta_star": delta_star, "wynik_star": _wiersz(delta_star, r_star, alpha),
            "c": c, "delta_zero": delta_zero}


def siatka_1d(model, Y2, brak, trt, delty, alpha=ALPHA) -> pd.DataFrame:
    return pd.DataFrame([_wiersz(float(d), beta1_mnar(model, Y2, brak, trt, float(d), alpha=alpha), alpha)
                         for d in delty])


def tipping_point_2d(model, Y2, brak, trt, d1_grid, d2_grid, alpha=ALPHA) -> dict:
    """Dwuwymiarowa siatka (δ1 – braki w TRT=1, δ2 – braki w TRT=2)."""
    P = np.empty((len(d2_grid), len(d1_grid)))
    E = np.empty_like(P)
    for a, d2 in enumerate(d2_grid):
        for b, d1 in enumerate(d1_grid):
            r = beta1_mnar(model, Y2, brak, trt, d1, d2, alpha=alpha)
            P[a, b], E[a, b] = r["p-value"], r["Oszacowanie"]
    return {"d1": np.asarray(d1_grid), "d2": np.asarray(d2_grid), "p": P, "beta1": E}


def krytyczne_delta1(model, Y2, brak, trt, d2_wartosci, kierunek, alpha=ALPHA,
                     zakres: float = 50.0) -> pd.DataFrame:
    """Dla zadanych δ2 – dokładne δ1*(δ2), przy którym p(β1) = α (metoda Brenta)."""
    wiersze = []
    for d2 in d2_wartosci:
        f = lambda d1: beta1_mnar(model, Y2, brak, trt, d1, d2, alpha)["p-value"] - alpha  # noqa: E731
        siatka = kierunek * np.arange(0, zakres + KROK_DELTA, KROK_DELTA) + 0.0
        wart = np.array([f(d) for d in siatka])
        zm = np.where(np.sign(wart[:-1]) != np.sign(wart[1:]))[0]
        if len(zm) == 0:
            wiersze.append({"δ2 (TRT=2)": d2, "δ1* (TRT=1)": np.nan})
            continue
        lo, hi = sorted([siatka[zm[0]], siatka[zm[0] + 1]])
        wiersze.append({"δ2 (TRT=2)": d2, "δ1* (TRT=1)": optimize.brentq(f, lo, hi, xtol=1e-10)})
    return pd.DataFrame(wiersze)


# =============================================================================
# 8. WYKRESY
# =============================================================================
def wykres_diagnostyka_imputacji(df, Y2, plik=None):
    """Rozkłady Y_T2 zaobserwowanych vs imputowanych (wszystkie m) w grupach TRT
    oraz zależność Y_T2 ~ Y_T1 dla 1. imputacji."""
    brak = df["brak_y_t2"].to_numpy()
    trt = df["trt"].to_numpy()
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for a, g in zip(ax[:2], [1, 2]):
        lab = "TRT=1 (czynnik)" if g == 1 else "TRT=2 (kontrola)"
        obs = df.loc[(~brak) & (trt == g), "y_t2"]
        imp = Y2[brak & (trt == g), :].ravel()
        sns.kdeplot(obs, ax=a, color=KOL_TRT[g], lw=2.2, label=f"zaobserwowane (n={len(obs)})")
        sns.kdeplot(imp, ax=a, color="black", lw=2, ls="--",
                    label=f"imputowane ({int((brak & (trt == g)).sum())} × m={Y2.shape[1]})")
        a.axvline(obs.mean(), color=KOL_TRT[g], lw=1, ls=":")
        a.axvline(imp.mean(), color="black", lw=1, ls=":")
        a.set_title(f"Y_T2: zaobserwowane vs imputowane – {lab}")
        a.set_xlabel("y_t2"); a.set_ylabel("Gęstość"); a.legend(fontsize=9)
    d = zbior_imputowany(df, Y2, 0)
    for g in [1, 2]:
        s = d["trt"] == g
        ax[2].scatter(d.loc[s & ~d["brak_y_t2"], "y_t1"], d.loc[s & ~d["brak_y_t2"], "y_t2"],
                      s=18, alpha=.35, color=KOL_TRT[g], label=f"TRT={g} zaobserwowane")
        ax[2].scatter(d.loc[s & d["brak_y_t2"], "y_t1"], d.loc[s & d["brak_y_t2"], "y_t2"],
                      s=26, alpha=.9, marker="x", color=KOL_TRT[g], label=f"TRT={g} imputowane")
    ax[2].set_title("Y_T2 ~ Y_T1 (imputacja nr 1)")
    ax[2].set_xlabel("y_t1"); ax[2].set_ylabel("y_t2"); ax[2].legend(fontsize=8)
    plt.tight_layout()
    if plik:
        fig.savefig(plik, dpi=DPI, bbox_inches="tight")
    return fig


def wykres_tipping_1d(siatka: pd.DataFrame, tp: dict, wsp_mar, alpha=ALPHA, plik=None):
    """β1(δ) z 95% CI oraz p-value(δ) z linią α i punktem δ*."""
    ds, ws = tp["delta_star"], tp["wynik_star"]
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.5))
    a = ax[0]
    a.fill_between(siatka["δ"], siatka["CI 2.5%"], siatka["CI 97.5%"], color="#2a6f97",
                   alpha=.18, label="95% CI (reguły Rubina)")
    a.plot(siatka["δ"], siatka["β1"], color="#2a6f97", lw=2.2, label=r"$\hat\beta_1(\delta)$")
    a.axhline(0, color="gray", ls="--", lw=1)
    a.axvline(ds, color="#c1121f", ls="--", lw=1.4)
    a.scatter([0], [wsp_mar.iloc[1]["Oszacowanie"]], color="black", zorder=5, s=45,
              label=f"MAR (δ=0): β1={wsp_mar.iloc[1]['Oszacowanie']:.3f}")
    a.scatter([ds], [ws["CI 2.5%"]], color="#c1121f", zorder=5, s=55,
              label=f"δ* = {ds:.3f} (dolna granica CI = 0)")
    a.set_xlabel(r"przesunięcie $\delta$ imputowanych Y_T2 w grupie TRT=1")
    a.set_ylabel(r"$\hat\beta_1$ (TRT=1 − TRT=2)")
    a.set_title(r"Efekt czynnika $\beta_1$ w funkcji przesunięcia $\delta$ (MNAR)")
    a.legend(fontsize=9, loc="best")

    a = ax[1]
    a.plot(siatka["δ"], siatka["p-value"], color="#2a6f97", lw=2.2)
    a.axhline(alpha, color="#c1121f", ls="--", lw=1.4, label=f"α = {alpha}")
    a.axvline(ds, color="#c1121f", ls=":", lw=1.2)
    a.scatter([ds], [alpha], color="#c1121f", s=55, zorder=5, label=f"δ* = {ds:.3f}")
    a.set_yscale("log")
    a.set_xlabel(r"przesunięcie $\delta$ imputowanych Y_T2 w grupie TRT=1")
    a.set_ylabel(r"p-value dla $\beta_1$ (skala log)")
    a.set_title(r"p-value($\delta$) – punkt krytyczny (tipping point)")
    lo, hi = a.get_ylim()
    sig = siatka["p-value"] < alpha
    a.fill_between(siatka["δ"], lo, hi, where=sig, color="#2a6f97", alpha=.06,
                   label="H0 odrzucona (p < α)")
    a.set_ylim(lo, hi)
    a.legend(fontsize=9, loc="best")
    plt.tight_layout()
    if plik:
        fig.savefig(plik, dpi=DPI, bbox_inches="tight")
    return fig


def wykres_tipping_2d(tp2: dict, tp1: dict, alpha=ALPHA, plik=None):
    """Mapa ciepła p-value i β1 na siatce (δ1, δ2) z granicą decyzyjną p = 0.05."""
    d1, d2, P, E = tp2["d1"], tp2["d2"], tp2["p"], tp2["beta1"]
    D1, D2 = np.meshgrid(d1, d2)
    fig, ax = plt.subplots(1, 2, figsize=(17, 7))

    a = ax[0]
    pc = a.pcolormesh(D1, D2, np.clip(P, 1e-12, 1), cmap="mako_r",
                      norm=LogNorm(vmin=1e-12, vmax=1), shading="auto")
    cb = fig.colorbar(pc, ax=a, pad=0.02)
    cb.set_label(r"p-value dla $\beta_1$ (skala log, obcięte do $10^{-12}$)")
    a.contour(D1, D2, P, levels=[alpha], colors="#ffb703", linewidths=2.8)
    a.set_title(r"p-value($\delta_1, \delta_2$) – granica decyzyjna p = 0.05 (linia żółta)")

    a = ax[1]
    lim = np.nanmax(np.abs(E))
    pc = a.pcolormesh(D1, D2, E, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim),
                      shading="auto")
    cb = fig.colorbar(pc, ax=a, pad=0.02)
    cb.set_label(r"$\hat\beta_1$ (TRT=1 − TRT=2)")
    a.contour(D1, D2, P, levels=[alpha], colors="black", linewidths=2.4)
    cs0 = a.contour(D1, D2, E, levels=[0], colors="black", linewidths=1.2, linestyles="--")
    a.clabel(cs0, fmt={0: "β1 = 0"}, fontsize=9)
    a.set_title(r"$\hat\beta_1(\delta_1, \delta_2)$ – linia ciągła: p = 0.05, przerywana: $\beta_1$ = 0")

    for a in ax:
        a.scatter([0], [0], marker="*", s=260, color="white", edgecolor="black", zorder=6,
                  label=r"MAR ($\delta_1 = \delta_2 = 0$)")
        a.scatter([tp1["delta_star"]], [0], marker="o", s=90, color="#c1121f", edgecolor="white",
                  zorder=6, label=f"δ* 1D = {tp1['delta_star']:.2f}")
        a.plot(d1, d1, color="#495057", ls=":", lw=1.4, label=r"$\delta_1 = \delta_2$ (jednakowe przesunięcie)")
        a.set_xlim(d1.min(), d1.max()); a.set_ylim(d2.min(), d2.max())
        a.set_xlabel(r"$\delta_1$ – przesunięcie imputowanych Y_T2 w grupie TRT=1")
        a.set_ylabel(r"$\delta_2$ – przesunięcie imputowanych Y_T2 w grupie TRT=2")
        a.legend(loc="lower right", fontsize=8.5, framealpha=.85)
        a.grid(False)
    fig.suptitle("Dwuwymiarowa analiza punktu krytycznego (2D tipping-point analysis, MNAR)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    if plik:
        fig.savefig(plik, dpi=DPI, bbox_inches="tight")
    return fig


def wykres_forest(wiersze: pd.DataFrame, plik=None):
    """Porównanie β1 (95% CI) między scenariuszami."""
    fig, ax = plt.subplots(figsize=(10, 0.75 * len(wiersze) + 1.5))
    y = np.arange(len(wiersze))[::-1]
    kol = ["#6c757d", "#2a6f97", "#5fa8d3", "#c1121f", "#e85d04"][: len(wiersze)]
    for yi, (_, r), c in zip(y, wiersze.iterrows(), kol):
        ax.errorbar(r["β1"], yi, xerr=[[r["β1"] - r["CI 2.5%"]], [r["CI 97.5%"] - r["β1"]]],
                    fmt="o", color=c, capsize=5, ms=8, lw=2)
        ax.text(r["CI 97.5%"] + 0.05, yi, f"{r['β1']:.3f} [{r['CI 2.5%']:.3f}; {r['CI 97.5%']:.3f}], "
                f"p={r['p-value']:.2g}", va="center", fontsize=9)
    ax.axvline(0, color="gray", ls="--")
    ax.set_yticks(y); ax.set_yticklabels(wiersze.index)
    ax.set_xlabel(r"$\hat\beta_1$ (TRT=1 − TRT=2) z 95% CI")
    ax.set_title("Efekt czynnika w modelu ANCOVA – porównanie scenariuszy braków danych")
    xmax = wiersze["CI 97.5%"].max()
    ax.set_xlim(min(-0.2, wiersze["CI 2.5%"].min() - 0.2), xmax + 1.6)
    plt.tight_layout()
    if plik:
        fig.savefig(plik, dpi=DPI, bbox_inches="tight")
    return fig


# =============================================================================
# 9. URUCHOMIENIE CAŁEJ ANALIZY (wydruk konsoli = załącznik do zadania)
# =============================================================================
def _naglowek(tekst: str):
    print("\n" + "=" * 100 + f"\n{tekst}\n" + "=" * 100)


def _tab(d: pd.DataFrame, nd: int = 4):
    print(d.round(nd).to_string())


class _Tee(io.TextIOBase):
    """Jednoczesny zapis na konsolę i do pliku."""

    def __init__(self, *strumienie):
        self.strumienie = strumienie

    def write(self, s):
        for st in self.strumienie:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.strumienie:
            st.flush()


def uruchom_analize() -> dict:
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)
    out = {}

    # ---------------------------------------------------------------- dane
    _naglowek("PUNKT 3 – ANCOVA przy założeniach MCAR / MAR oraz analiza wrażliwości MNAR")
    csv_lbl = CSV_FILE.relative_to(ROOT) if CSV_FILE.is_relative_to(ROOT) else CSV_FILE
    print(f"Plik danych: {csv_lbl}   |   SEED = {SEED}   |   m = {M_IMP}   |   α = {ALPHA}")
    df = wczytaj_dane(CSV_FILE)
    brak = df["brak_y_t2"].to_numpy()
    trt = df["trt"].to_numpy()
    print(f"Wymiary zbioru: {df.shape[0]} obserwacji; zmienne: y_t1, x1, trt, x2, y_t2")
    print("\nBraki danych (zgodnie z pkt 1 – wyłącznie Y_T2):")
    _tab(opis_brakow(df), 2)
    print("\nBraki Y_T2 wg TRT:")
    _tab(df.groupby("trt")["brak_y_t2"].agg(n="size", braki="sum", odsetek="mean"), 4)
    print("\nModel analityczny: chg ~ C(trt, Treatment(reference=2)) + y_t1,  chg = y_t2 − y_t1")
    print("β1 = różnica średnich CHG (TRT=1 − TRT=2) skorygowana o Y_T1; H0: β1 = 0.")

    # ---------------------------------------------------------------- MCAR
    _naglowek("A. SCENARIUSZ MCAR – analiza pełnych przypadków (complete-case, OLS)")
    mcar = analiza_mcar(df)
    out["mcar"] = mcar
    print(mcar["model"].summary())
    print(f"\nN = {mcar['n']},  R² = {mcar['r2']:.4f}")
    print("\nWspółczynniki ANCOVA (CC):")
    _tab(mcar["wsp"])
    print(f"\nŚrednie CHG (surowe oraz LS-means przy Y_T1 = {mcar['y1_ref']:.4f}, średnia z n = {mcar['n']} obserwacji kompletnych):")
    _tab(mcar["srednie"])

    # ---------------------------------------------------------------- MAR
    _naglowek(f"B. SCENARIUSZ MAR – imputacja wielokrotna FCS/MICE (m = {M_IMP}) + reguły Rubina")
    print("Model imputacyjny (osobno w warstwach TRT = 1 i TRT = 2, metoda 'norm' – bayesowska regresja liniowa):")
    print("  y_t2 | TRT=g  ~  γ0g + γ1g·y_t1 + γ2g·x1 + γ3g·x2 + e,   e ~ N(0, σ²_g),   g ∈ {1, 2}")
    print("  σ*² = SSR/χ²(n_g − 4),  γ* = γ̂ + σ*·chol((X'X)⁻¹)·z,  y_imp = Xγ* + σ*·ε")
    print("  CHG liczone pasywnie po imputacji: chg = y_t2_imp − y_t1.")
    Y2 = wielokrotna_imputacja(df)
    out["Y2"] = Y2

    # parametry modeli imputacyjnych (dopasowanie na obserwacjach kompletnych) – do raportu
    print("\nOszacowania modeli imputacyjnych (OLS na danych zaobserwowanych, punkt startowy losowania):")
    for g in [1, 2]:
        dg = df.loc[(df["trt"] == g) & ~df["brak_y_t2"]]
        rg = smf.ols("y_t2 ~ y_t1 + x1 + x2", data=dg).fit()
        tab_g = pd.DataFrame({"γ̂": rg.params, "SE": rg.bse, "p-value": rg.pvalues})
        print(f"\n  TRT={g}: n_obs = {int(rg.nobs)}, σ̂ = {np.sqrt(rg.mse_resid):.4f}, R² = {rg.rsquared:.4f}")
        _tab(tab_g)

    mar = analiza_mar(df, Y2)
    out["mar"] = mar
    print("\nWspółczynniki ANCOVA (MI, reguły Rubina; ν_com = n − 3 = 802):")
    _tab(mar["wsp"])
    print(f"\nN = {mar['n']} (wszystkie obserwacje),  R² (łączone, Fisher z) = {mar['r2']:.4f},  "
          f"RMSE (średnie σ̂) = {mar['rmse']:.4f}")
    print("\nŚrednie CHG (surowe i LS-means), reguły Rubina:")
    _tab(mar["srednie"])

    # weryfikacja krzyżowa – scikit-learn IterativeImputer
    print("\nWeryfikacja krzyżowa: IterativeImputer(BayesianRidge, sample_posterior=True), warstwy TRT, m = "
          f"{M_IMP}:")
    Y2_sk = wielokrotna_imputacja_sklearn(df)
    mar_sk = analiza_mar(df, Y2_sk)
    out["mar_sk"] = mar_sk
    _tab(mar_sk["wsp"][["Oszacowanie", "SE", "df (ν_adj)", "p-value", "CI 2.5%", "CI 97.5%", "FMI"]])

    # ---------------------------------------------------------------- weryfikacja
    _naglowek("C. WERYFIKACJA POPRAWNOŚCI")
    model = SzybkaANCOVA(df)
    B_fast, V_fast = model.dopasuj(Y2)
    obs_y2 = df.loc[~brak, "y_t2"].to_numpy()
    kontrole = {
        "Brak NaN po imputacji (wszystkie m zbiorów)": not np.isnan(Y2).any(),
        "Wartości zaobserwowane Y_T2 identyczne we wszystkich imputacjach":
            bool(np.all(Y2[~brak, :] == obs_y2[:, None])),
        "Imputacje różnią się między zbiorami (B > 0)": bool(np.all(Y2[brak, :].std(axis=1) > 0)),
        "Szybka ANCOVA = statsmodels (β i Var(β))":
            bool(np.allclose(B_fast, mar["Q_beta"]) and np.allclose(V_fast, mar["U_beta"])),
    }
    Y_test = przesun_mnar(Y2, brak, trt, -3.0, 1.5)
    kontrole["MNAR: wartości zaobserwowane nietknięte po przesunięciu δ"] = bool(
        np.all(Y_test[~brak, :] == Y2[~brak, :]))
    kontrole["MNAR: imputowane TRT=1 przesunięte dokładnie o δ1, TRT=2 o δ2"] = bool(
        np.allclose(Y_test[brak & (trt == 1)] - Y2[brak & (trt == 1)], -3.0)
        and np.allclose(Y_test[brak & (trt == 2)] - Y2[brak & (trt == 2)], 1.5))
    for k_, v_ in kontrole.items():
        print(f"  [{'OK' if v_ else 'BŁĄD'}] {k_}")
    assert all(kontrole.values()), "Weryfikacja nie powiodła się!"

    # ---------------------------------------------------------------- MNAR 1D
    _naglowek("D. SCENARIUSZ MNAR – δ-adjustment (przesunięcie imputowanych Y_T2 w grupie TRT=1)")
    tp = tipping_point_1d(model, Y2, brak, trt, mar["wsp"])
    out["tp1"] = tp
    kier = "ujemnym (w stronę utraty istotności)" if tp["kierunek"] < 0 else "dodatnim"
    print(f"β1 (MAR) = {mar['wsp'].iloc[1]['Oszacowanie']:.4f}, p = {mar['wsp'].iloc[1]['p-value']:.3g} → "
          f"{'istotny' if tp['istotny_mar'] else 'nieistotny'}; siatkę δ przesuwamy w kierunku {kier}.")
    print(f"Y_T2_MNAR = Y_T2_MAR + δ wyłącznie dla {int((brak & (trt == 1)).sum())} imputowanych obserwacji "
          f"z TRT=1; zgrubna siatka co {KROK_DELTA}.")
    print("\nZgrubna siatka δ (rozszerzana automatycznie do zmiany decyzji + 2 punkty dalej):")
    _tab(tp["zgrubna"])
    print("\nZagęszczona siatka w przedziale przejścia (krok = {:.2f}):".format(KROK_DELTA / 10))
    _tab(tp["gesta"])
    ws = tp["wynik_star"]
    print(f"\n>>> PUNKT KRYTYCZNY (Brent, p(δ*) = α): δ* = {tp['delta_star']:.4f}")
    print(f"    β1(δ*) = {ws['β1']:.4f}, SE = {ws['SE']:.4f}, 95% CI = [{ws['CI 2.5%']:.4f}; {ws['CI 97.5%']:.4f}], "
          f"p = {ws['p-value']:.6f}")
    print(f"    β1(δ) = β1_MAR + c·δ,  c = {tp['c']:.4f}  →  δ dla β1 = 0: δ0 = {tp['delta_zero']:.4f}")
    p_lo = beta1_mnar(model, Y2, brak, trt, tp["delta_star"] - tp["kierunek"] * 1e-3)["p-value"]
    p_hi = beta1_mnar(model, Y2, brak, trt, tp["delta_star"] + tp["kierunek"] * 1e-3)["p-value"]
    print(f"    Kontrola: p(δ* ∓ 0.001) = {p_lo:.5f} / {p_hi:.5f}  (próg α = {ALPHA} przecięty: "
          f"{'TAK' if (p_lo < ALPHA) != (p_hi < ALPHA) else 'NIE'})")
    assert (p_lo < ALPHA) != (p_hi < ALPHA)

    # skala δ* – ocena realizmu scenariusza
    sd_y2 = df["y_t2"].std(); sd_y2_1 = df.loc[df["trt"] == 1, "y_t2"].std()
    sd_chg = df["chg"].std(); sd_chg_1 = df.loc[df["trt"] == 1, "chg"].std()
    ds = tp["delta_star"]
    chg_imp_1 = (Y2[brak & (trt == 1)] - df.loc[brak & (trt == 1), "y_t1"].to_numpy()[:, None]).mean()
    chg_obs_1 = df.loc[(trt == 1) & ~brak, "chg"].mean()
    chg_obs_2 = df.loc[(trt == 2) & ~brak, "chg"].mean()
    chg_mar_2 = mar["srednie"].loc["CHG surowa – TRT=2", "Oszacowanie"]
    delta_ref = chg_mar_2 - chg_imp_1
    skala = pd.DataFrame({
        "wartość": [ds, sd_y2, sd_y2_1, sd_chg, sd_chg_1, mar["rmse"], ds / sd_y2, ds / sd_y2_1,
                    ds / sd_chg, ds / sd_chg_1, ds / mar["rmse"], chg_obs_1, chg_imp_1, chg_imp_1 + ds,
                    chg_obs_2, chg_mar_2, delta_ref],
    }, index=["δ*", "SD(Y_T2) obs., cała próba", "SD(Y_T2) obs., TRT=1", "SD(CHG) obs., cała próba",
              "SD(CHG) obs., TRT=1", "RMSE ANCOVA (MAR)", "δ* / SD(Y_T2)", "δ* / SD(Y_T2 | TRT=1)",
              "δ* / SD(CHG)", "δ* / SD(CHG | TRT=1)", "δ* / RMSE",
              "Śr. CHG zaobserwowana, TRT=1", "Śr. CHG imputowana (MAR), braki TRT=1",
              "Śr. CHG imputowana przy δ*, braki TRT=1", "Śr. CHG zaobserwowana, TRT=2",
              "Śr. CHG (MAR), TRT=2 – cała grupa",
              "δ_ref: braki TRT=1 → średnia CHG grupy kontrolnej"])
    out["skala"] = skala
    print("\nSkala punktu krytycznego (ocena realizmu scenariusza MNAR):")
    _tab(skala)

    delty_plot = np.round(np.arange(min(0.0, 1.6 * ds, 1.1 * tp["delta_zero"]) - 0.5,
                                    max(0.0, 1.6 * ds, 1.1 * tp["delta_zero"]) + 0.5 + 1e-9, 0.05), 4)
    siatka = siatka_1d(model, Y2, brak, trt, delty_plot)
    out["siatka_1d"] = siatka

    # ---------------------------------------------------------------- MNAR 2D
    _naglowek("E. MNAR – DWUWYMIAROWA ANALIZA PUNKTU KRYTYCZNEGO (δ1: braki TRT=1, δ2: braki TRT=2)")
    R = float(np.ceil(1.5 * abs(ds)))
    grid = np.round(np.linspace(-R, R, 121), 6)
    tp2 = tipping_point_2d(model, Y2, brak, trt, grid, grid)
    out["tp2"] = tp2
    print(f"Siatka: δ1, δ2 ∈ [{-R:.0f}; {R:.0f}], 121 × 121 = {121 * 121} kombinacji, każda: m = {M_IMP} "
          "modeli ANCOVA + reguły Rubina.")
    d2_tab = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    kryt = krytyczne_delta1(model, Y2, brak, trt, d2_tab, tp["kierunek"])
    out["kryt_2d"] = kryt
    print("\nKrytyczne δ1*(δ2) – granica p = 0.05 dla wybranych przesunięć w grupie kontrolnej:")
    _tab(kryt)
    wybrane = [(-2, 0), (-4, 0), (-4, 2), (-2, 2), (0, 2), (0, 4), (-6, 0), (-4, -2), (2, 2), (-3, 3)]
    tab2 = pd.DataFrame([{**{"δ1 (TRT=1)": a, "δ2 (TRT=2)": b},
                          **{k: v for k, v in _wiersz(np.nan, beta1_mnar(model, Y2, brak, trt, a, b)).items()
                             if k != "δ"}} for a, b in wybrane])
    out["tab_2d"] = tab2
    print("\nWybrane kombinacje (δ1, δ2):")
    _tab(tab2)

    # ---------------------------------------------------------------- tabele porównawcze
    _naglowek("F. TABELA PORÓWNAWCZA – MCAR (CC) vs MAR (MI) vs MNAR (δ*)")
    comp = pd.DataFrame({
        "MCAR (CC)": [mcar["n"], mcar["srednie"].iloc[0, 0], mcar["srednie"].iloc[1, 0],
                      mcar["srednie"].iloc[2, 0], mcar["srednie"].iloc[3, 0],
                      *mcar["wsp"].iloc[1][["Oszacowanie", "SE", "CI 2.5%", "CI 97.5%", "p-value"]],
                      mcar["wsp"].iloc[2]["Oszacowanie"], mcar["r2"], np.nan, np.nan],
        "MAR (MI, FCS norm)": [mar["n"], *mar["srednie"]["Oszacowanie"],
                               *mar["wsp"].iloc[1][["Oszacowanie", "SE", "CI 2.5%", "CI 97.5%", "p-value"]],
                               mar["wsp"].iloc[2]["Oszacowanie"], mar["r2"], mar["wsp"].iloc[1]["RIV"],
                               mar["wsp"].iloc[1]["FMI"]],
        "MAR (MI, sklearn)": [mar_sk["n"], *mar_sk["srednie"]["Oszacowanie"],
                              *mar_sk["wsp"].iloc[1][["Oszacowanie", "SE", "CI 2.5%", "CI 97.5%", "p-value"]],
                              mar_sk["wsp"].iloc[2]["Oszacowanie"], mar_sk["r2"], mar_sk["wsp"].iloc[1]["RIV"],
                              mar_sk["wsp"].iloc[1]["FMI"]],
    }, index=["N", "Śr. CHG surowa TRT=1", "Śr. CHG surowa TRT=2", "LS-mean CHG TRT=1",
              "LS-mean CHG TRT=2", "β1", "SE(β1)", "CI 2.5%", "CI 97.5%", "p-value", "β2 (Y_T1)", "R²",
              "RIV(β1)", "FMI(β1)"])
    out["porownanie"] = comp
    _tab(comp)

    # ---------------------------------------------------------------- zapis
    _naglowek("G. ZAPIS WYNIKÓW")
    mcar["wsp"].to_csv(RES / "tab_p3_ancova_mcar.csv", encoding="utf-8-sig")
    mar["wsp"].to_csv(RES / "tab_p3_ancova_mar.csv", encoding="utf-8-sig")
    pd.concat({"MCAR": mcar["srednie"], "MAR": mar["srednie"]}).to_csv(
        RES / "tab_p3_srednie_chg.csv", encoding="utf-8-sig")
    comp.to_csv(RES / "tab_p3_porownanie.csv", encoding="utf-8-sig")
    pd.concat([tp["zgrubna"], tp["gesta"], pd.DataFrame([ws])], keys=["zgrubna", "gesta", "delta_star"]).to_csv(
        RES / "tab_p3_tipping_1d.csv", encoding="utf-8-sig")
    siatka.to_csv(RES / "tab_p3_tipping_1d_gesta_siatka.csv", encoding="utf-8-sig", index=False)
    D1, D2 = np.meshgrid(tp2["d1"], tp2["d2"])
    pd.DataFrame({"δ1": D1.ravel(), "δ2": D2.ravel(), "β1": tp2["beta1"].ravel(),
                  "p-value": tp2["p"].ravel()}).to_csv(RES / "tab_p3_tipping_2d.csv", encoding="utf-8-sig",
                                                       index=False)
    kryt.to_csv(RES / "tab_p3_tipping_2d_krytyczne.csv", encoding="utf-8-sig", index=False)
    skala.to_csv(RES / "tab_p3_skala_delta.csv", encoding="utf-8-sig")

    wykres_diagnostyka_imputacji(df, Y2, FIG / "fig6_imputacja_diagnostyka.png")
    wykres_tipping_1d(siatka, tp, mar["wsp"], plik=FIG / "fig7_tipping_point_1d.png")
    wykres_tipping_2d(tp2, tp, plik=FIG / "fig8_tipping_point_2d.png")
    forest = pd.DataFrame([
        {"β1": mcar["wsp"].iloc[1]["Oszacowanie"], "CI 2.5%": mcar["wsp"].iloc[1]["CI 2.5%"],
         "CI 97.5%": mcar["wsp"].iloc[1]["CI 97.5%"], "p-value": mcar["wsp"].iloc[1]["p-value"]},
        {"β1": mar["wsp"].iloc[1]["Oszacowanie"], "CI 2.5%": mar["wsp"].iloc[1]["CI 2.5%"],
         "CI 97.5%": mar["wsp"].iloc[1]["CI 97.5%"], "p-value": mar["wsp"].iloc[1]["p-value"]},
        {"β1": mar_sk["wsp"].iloc[1]["Oszacowanie"], "CI 2.5%": mar_sk["wsp"].iloc[1]["CI 2.5%"],
         "CI 97.5%": mar_sk["wsp"].iloc[1]["CI 97.5%"], "p-value": mar_sk["wsp"].iloc[1]["p-value"]},
        {k: tp["wynik_star"][k] for k in ["β1", "CI 2.5%", "CI 97.5%", "p-value"]},
    ], index=["MCAR – CC", f"MAR – MI FCS norm (m={M_IMP})", f"MAR – MI sklearn (m={M_IMP})",
              f"MNAR – δ* = {ds:.2f} (TRT=1)"])
    wykres_forest(forest, FIG / "fig9_forest_beta1.png")
    plt.close("all")
    for p in sorted(RES.glob("tab_p3_*.csv")) + sorted(FIG.glob("fig[6-9]_*.png")):
        print(f"  zapisano: {p.relative_to(ROOT)}")
    return out


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    RES.mkdir(exist_ok=True)
    plik_txt = RES / "punkt3_wydruk_konsoli.txt"
    stary = sys.stdout
    with open(plik_txt, "w", encoding="utf-8") as fh:
        sys.stdout = _Tee(stary, fh)
        try:
            with warnings.catch_warnings(record=True) as ostrz:
                warnings.simplefilter("always")
                uruchom_analize()
            _naglowek("KONIEC – podsumowanie ostrzeżeń")
            print(f"Liczba ostrzeżeń zarejestrowanych podczas analizy: {len(ostrz)}")
            for w in ostrz:
                print(f"  {w.category.__name__}: {w.message}  ({Path(w.filename).name}:{w.lineno})")
            print(f"Wydruk zapisano w: {plik_txt.relative_to(ROOT)}")
        finally:
            sys.stdout = stary


if __name__ == "__main__":
    main()
