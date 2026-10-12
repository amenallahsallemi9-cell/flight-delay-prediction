"""Métriques et graphiques pour le projet de prédiction des retards d'avion."""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
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
