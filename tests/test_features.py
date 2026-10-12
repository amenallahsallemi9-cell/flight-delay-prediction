"""Tests des fonctions de src/ : absence de fuite, codes d'aéroports, séparation temporelle.

Les tests utilisent de petites données synthétiques : ils tournent en quelques secondes
sans télécharger le jeu Kaggle.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import evaluation as E  # noqa: E402
import features as F  # noqa: E402


@pytest.fixture
def vols():
    """~20 000 vols synthétiques répartis sur toute l'année 2015."""
    rng = np.random.default_rng(0)
    n = 20_000
    dates = pd.Timestamp("2015-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    df = pd.DataFrame({
        "DATE": dates,
        "MONTH": dates.month.astype("int8"),
        "DAY": dates.day.astype("int8"),
        "AIRLINE": pd.Categorical(rng.choice(["AA", "DL", "NK"], n)),
        "ORIGIN_AIRPORT": pd.Categorical(rng.choice(["JFK", "LAX", "ORD", "ATL"], n)),
        "DESTINATION_AIRPORT": pd.Categorical(rng.choice(["SFO", "SEA", "BOS"], n)),
        "RETARD": rng.integers(0, 2, n).astype("int8"),
    })
    return df.sort_values("DATE").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fuite de données
# ---------------------------------------------------------------------------
def test_aucune_colonne_interdite_lue():
    """Seules la cible et les indicateurs de filtrage peuvent être des colonnes « du jour du vol »."""
    autorisees = {"ARRIVAL_DELAY", "CANCELLED", "DIVERTED"}
    assert set(F.COLONNES_LUES) & set(F.COLONNES_INTERDITES) == autorisees


def test_variables_du_modele_sans_fuite():
    variables = F.CAT + F.NUM_BASE + F.NUM_HIST
    assert not set(variables) & set(F.COLONNES_INTERDITES)


def test_taux_historiques_ne_dependent_pas_du_futur(vols):
    """Modifier les cibles à partir du 1er juin ne change aucun taux calculé avant cette date."""
    ref = F.ajouter_taux_historiques(vols, F.DEBUT_TEST)
    perturbe = vols.copy()
    futur = perturbe["DATE"] >= "2015-06-01"
    perturbe.loc[futur, "RETARD"] = 1 - perturbe.loc[futur, "RETARD"]
    res = F.ajouter_taux_historiques(perturbe, F.DEBUT_TEST)
    np.testing.assert_allclose(res.loc[~futur, F.NUM_HIST], ref.loc[~futur, F.NUM_HIST])


def test_taux_historiques_excluent_le_jour_meme(vols):
    """Les vols d'un même jour ne voient pas les résultats des autres vols de ce jour."""
    jour = vols["DATE"] == "2015-03-10"
    ref = F.ajouter_taux_historiques(vols, F.DEBUT_TEST)
    perturbe = vols.copy()
    perturbe.loc[jour, "RETARD"] = 1
    res = F.ajouter_taux_historiques(perturbe, F.DEBUT_TEST)
    np.testing.assert_allclose(res.loc[jour, F.NUM_HIST], ref.loc[jour, F.NUM_HIST])


def test_taux_figes_apres_la_coupure(vols):
    """Après la coupure, les taux ne dépendent que de l'entraînement, pas des cibles du test."""
    test = vols["DATE"] >= F.DEBUT_TEST
    ref = F.ajouter_taux_historiques(vols, F.DEBUT_TEST)
    perturbe = vols.copy()
    perturbe.loc[test, "RETARD"] = 1
    res = F.ajouter_taux_historiques(perturbe, F.DEBUT_TEST)
    np.testing.assert_allclose(res.loc[test, F.NUM_HIST], ref.loc[test, F.NUM_HIST])
    # Un taux figé est constant pour une même compagnie
    assert ref.loc[test].groupby("AIRLINE", observed=True)["TAUX_HIST_COMPAGNIE"].nunique().eq(1).all()


def test_lissage_et_categorie_inconnue():
    """Formule de lissage vérifiée à la main ; une catégorie jamais vue reçoit le taux global."""
    df = pd.DataFrame({
        "DATE": pd.to_datetime(["2015-01-01"] * 4 + ["2015-02-01"] * 2),
        "AIRLINE": pd.Categorical(["AA", "AA", "DL", "DL", "AA", "NK"]),
        "ORIGIN_AIRPORT": pd.Categorical(["JFK"] * 6), "DESTINATION_AIRPORT": pd.Categorical(["LAX"] * 6),
        "RETARD": np.array([1, 1, 0, 0, 0, 0], dtype="int8"),
    })
    res = F.ajouter_taux_historiques(df, "2015-01-15", m=2)
    # Taux global avant la coupure = 2/4 = 0,5 ; AA : 2 retards sur 2 → (2 + 2×0,5) / (2 + 2) = 0,75
    assert res.loc[4, "TAUX_HIST_COMPAGNIE"] == pytest.approx(0.75)
    assert res.loc[5, "TAUX_HIST_COMPAGNIE"] == pytest.approx(0.5)  # NK jamais vue → prior
    # Premier jour : aucun historique → valeur manquante
    assert res.loc[:3, "TAUX_HIST_COMPAGNIE"].isna().all()


# ---------------------------------------------------------------------------
# Codes d'aéroports d'octobre
# ---------------------------------------------------------------------------
def test_est_code_iata():
    s = pd.Series(["JFK", "12478", "LA", "lax", None, "ORD"])
    assert F.est_code_iata(s).tolist() == [True, False, False, False, False, True]


def _programme(mois, origine, destination, numero, heure, distance, n=5):
    return pd.DataFrame({
        "MONTH": [mois] * n, "AIRLINE": ["AA"] * n, "FLIGHT_NUMBER": [numero] * n,
        "SCHEDULED_DEPARTURE": [heure] * n, "DISTANCE": [distance] * n,
        "ORIGIN_AIRPORT": [origine] * n, "DESTINATION_AIRPORT": [destination] * n,
    })


def test_correspondance_octobre():
    df = pd.concat([
        _programme(9, "JFK", "LAX", 1, 700, 2475), _programme(11, "JFK", "LAX", 1, 700, 2475),
        _programme(10, "12478", "12892", 1, 700, 2475),
        _programme(9, "ORD", "ATL", 2, 900, 606), _programme(10, "13930", "10397", 2, 900, 606),
        _programme(10, "99999", "12478", 3, 1200, 1000),  # vol sans équivalent en sept./nov.
    ], ignore_index=True)
    corresp = F.construire_correspondance_octobre(df)
    assert corresp == {"12478": "JFK", "12892": "LAX", "13930": "ORD", "10397": "ATL"}

    corrige, _ = F.corriger_codes_octobre(df)
    assert F.est_code_iata(corrige["ORIGIN_AIRPORT"]).all()
    assert F.est_code_iata(corrige["DESTINATION_AIRPORT"]).all()
    assert len(corrige) == len(df) - 5  # seuls les vols avec un code inconnu (99999) sont retirés


def test_correspondance_octobre_ecarte_les_votes_minoritaires():
    """Un code dont les votes sont trop partagés (< 90 % pour le meilleur) n'est pas gardé."""
    df = pd.concat([
        _programme(9, "JFK", "LAX", 1, 700, 2475, n=1), _programme(10, "12478", "12892", 1, 700, 2475, n=1),
        _programme(9, "BOS", "LAX", 2, 800, 2611, n=1), _programme(10, "12478", "12892", 2, 800, 2611, n=1),
    ], ignore_index=True)
    corresp = F.construire_correspondance_octobre(df)
    assert "12478" not in corresp and corresp.get("12892") == "LAX"


def test_trafic_compte_les_departs_de_la_meme_heure():
    df = pd.DataFrame({
        "ORIGIN_AIRPORT": ["JFK", "JFK", "JFK", "LAX"], "MONTH": [1, 1, 1, 1], "DAY": [5, 5, 5, 5],
        "SCHEDULED_DEPARTURE": [705, 755, 805, 705],
    })
    assert F.ajouter_trafic(df)["TRAFIC_HEURE"].tolist() == [2, 2, 1, 1]


# ---------------------------------------------------------------------------
# Séparation temporelle
# ---------------------------------------------------------------------------
def test_separation_temporelle(vols):
    sep = F.separer_temps(vols)
    tr, val, test = (vols.loc[sep[k], "DATE"] for k in ["entrainement", "validation", "test"])
    # Ensembles disjoints et ordonnés dans le temps
    assert not (sep["entrainement"] & sep["validation"]).any()
    assert not ((sep["entrainement"] | sep["validation"]) & sep["test"]).any()
    assert tr.max() < val.min() and val.max() < test.min()
    # Mois attendus : janvier-juillet / août / septembre, novembre, décembre (pas octobre)
    assert set(tr.dt.month) == set(range(1, 8))
    assert set(val.dt.month) == {8}
    assert set(test.dt.month) == {9, 11, 12}


def test_aeroports_rares_choisis_sur_l_entrainement(vols):
    """Un aéroport absent de l'entraînement est classé « AUTRE », même s'il est fréquent au test."""
    vols = vols.copy()
    vols["ORIGIN_AIRPORT"] = vols["ORIGIN_AIRPORT"].cat.add_categories("MIA")
    vols.loc[vols["MONTH"] >= 9, "ORIGIN_AIRPORT"] = "MIA"
    res = F.regrouper_aeroports_rares(vols, F.separer_temps(vols)["entrainement"], top=10)
    assert "MIA" not in res["ORIGIN_AIRPORT"].cat.categories
    assert (res.loc[vols["MONTH"] >= 9, "ORIGIN_AIRPORT"] == "AUTRE").all()


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------
def test_metriques_et_seuil():
    y = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    seuil = E.choisir_seuil(y, proba)
    m = E.metriques(y, proba, seuil)
    assert m["ROC-AUC"] == pytest.approx(1.0) and m["F1"] == pytest.approx(1.0)
    assert 0.3 < seuil <= 0.8
