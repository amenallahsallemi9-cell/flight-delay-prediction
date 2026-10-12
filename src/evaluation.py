"""Métriques et graphiques pour le projet de prédiction des retards d'avion."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Palette sobre : une couleur d'accent, du gris pour les repères
BLEU, ORANGE, GRIS, ENCRE = "#2a78d6", "#eb6834", "#8a8984", "#52514e"
DOSSIER_FIGURES = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")


def style_graphiques() -> None:
    """Style matplotlib commun : grille discrète, pas de cadre superflu."""
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": GRIS, "axes.labelcolor": ENCRE, "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
        "axes.axisbelow": True, "xtick.color": ENCRE, "ytick.color": ENCRE,
        "font.size": 10, "legend.frameon": False,
    })


def sauver(fig, nom: str) -> None:
    """Exporte une figure dans docs/figures/ (utilisée par le README)."""
    os.makedirs(DOSSIER_FIGURES, exist_ok=True)
    fig.savefig(os.path.join(DOSSIER_FIGURES, f"{nom}.png"))


# ---------------------------------------------------------------------------
# Analyse exploratoire
# ---------------------------------------------------------------------------
def taux_retard(df: pd.DataFrame, col: str, min_vols: int = 0) -> pd.DataFrame:
    """Taux de retard (en %) et nombre de vols par modalité de `col`."""
    t = df.groupby(col, observed=True)["RETARD"].agg(taux="mean", vols="size")
    t["taux"] *= 100
    return t[t["vols"] >= min_vols]


def graphique_taux(serie: pd.Series, titre: str, xlabel: str, moyenne: float,
                   kind: str = "bar", ax=None, etiquettes: bool = True):
    """Taux de retard par catégorie, avec la moyenne globale en pointillés.

    kind = "bar" (barres verticales), "barh" (horizontales) ou "line" (courbe).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.6) if kind != "barh" else (7, 0.32 * len(serie) + 1))
    x = [str(i) for i in serie.index]
    if kind == "line":
        ax.plot(serie.index, serie.values, color=BLEU, lw=2, marker="o", ms=5)
    elif kind == "bar":
        ax.bar(x, serie.values, color=BLEU, width=0.7)
    else:
        ax.barh(x, serie.values, color=BLEU, height=0.7)
    ref = ax.axhline if kind != "barh" else ax.axvline
    ref(moyenne, color=GRIS, ls="--", lw=1.2, label=f"moyenne ({moyenne:.1f} %)")
    if kind == "barh":
        ax.set(title=titre, xlabel="% de vols en retard (> 15 min)", ylabel=xlabel)
        ax.grid(axis="y", visible=False)
        if etiquettes:
            for i, v in enumerate(serie.values):
                ax.text(v + 0.3, i, f"{v:.1f}", va="center", fontsize=8, color=ENCRE)
    else:
        ax.set(title=titre, xlabel=xlabel, ylabel="% de vols en retard (> 15 min)")
        ax.grid(axis="x", visible=False)
    ax.legend(loc="best")
    return ax


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------
def choisir_seuil(y_val, proba_val) -> float:
    """Seuil qui maximise le F1-score sur la **validation** (jamais sur le test)."""
    from sklearn.metrics import precision_recall_curve
    p, r, s = precision_recall_curve(y_val, proba_val)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    return float(s[np.argmax(f1[:-1])])


def metriques(y, proba, seuil: float) -> dict[str, float]:
    """ROC-AUC et PR-AUC (indépendantes du seuil), puis précision / rappel / F1 au seuil."""
    from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    pred = (np.asarray(proba) >= seuil).astype(int)
    return {
        "ROC-AUC": roc_auc_score(y, proba),
        "PR-AUC": average_precision_score(y, proba),
        "Précision": precision_score(y, pred, zero_division=0),
        "Rappel": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "Seuil": seuil,
    }


def courbes_roc_pr(y, probas: dict, titre_suffixe: str = ""):
    """Courbes ROC et précision-rappel de plusieurs modèles côte à côte."""
    from sklearn.metrics import (average_precision_score, precision_recall_curve, roc_auc_score,
                                 roc_curve)
    couleurs = [GRIS, ORANGE, BLEU, "#1baf7a"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for (nom, p), c in zip(probas.items(), couleurs):
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(fpr, tpr, color=c, lw=2, label=f"{nom} (AUC = {roc_auc_score(y, p):.3f})")
        prec, rec, _ = precision_recall_curve(y, p)
        a2.plot(rec, prec, color=c, lw=2, label=f"{nom} (AP = {average_precision_score(y, p):.3f})")
    a1.plot([0, 1], [0, 1], color=GRIS, ls=":", lw=1)
    a2.axhline(np.mean(y), color=GRIS, ls=":", lw=1, label=f"hasard ({np.mean(y):.3f})")
    a1.set(title=f"Courbe ROC{titre_suffixe}", xlabel="taux de faux positifs", ylabel="taux de vrais positifs (rappel)")
    a2.set(title=f"Courbe précision-rappel{titre_suffixe}", xlabel="rappel", ylabel="précision", ylim=(0, 1))
    a1.legend(loc="lower right"); a2.legend(loc="upper right")
    plt.tight_layout()
    return fig


def erreurs_par_segment(df: pd.DataFrame, y, proba, seuil: float, col: str,
                        min_vols: int = 2000) -> pd.DataFrame:
    """Qualité du modèle par segment : taux d'erreur, retards manqués, fausses alertes, ROC-AUC."""
    from sklearn.metrics import roc_auc_score
    d = pd.DataFrame({"seg": df[col].to_numpy(), "y": np.asarray(y), "p": np.asarray(proba)})
    d["pred"] = (d["p"] >= seuil).astype(int)

    def resume(g):
        retards, a_l_heure = g["y"] == 1, g["y"] == 0
        return pd.Series({
            "vols": len(g),
            "taux_retard_%": 100 * g["y"].mean(),
            "proba_moyenne_%": 100 * g["p"].mean(),
            "erreurs_%": 100 * (g["pred"] != g["y"]).mean(),
            "retards_manqués_%": 100 * (g.loc[retards, "pred"] == 0).mean() if retards.any() else np.nan,
            "fausses_alertes_%": 100 * (g.loc[a_l_heure, "pred"] == 1).mean() if a_l_heure.any() else np.nan,
            "ROC-AUC": roc_auc_score(g["y"], g["p"]) if g["y"].nunique() == 2 else np.nan,
        })

    res = d.groupby("seg", observed=True)[["y", "p", "pred"]].apply(resume)
    return res[res["vols"] >= min_vols].round(3)
