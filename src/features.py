"""Préparation des données et feature engineering pour la prédiction des retards d'avion.

Règle d'or de tout ce module : une variable n'est utilisable que si elle est connue
**au moment de la réservation**. Tout ce qui se mesure le jour du vol (retard au départ,
causes du retard, temps de roulage...) est une fuite de données (data leakage).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

SEUIL_RETARD = 15  # minutes : définition officielle d'un vol « en retard » aux États-Unis

# Colonnes lues dans flights.csv : uniquement des informations connues à l'avance,
# plus la cible (ARRIVAL_DELAY) et les indicateurs d'annulation / déroutement pour filtrer.
COLONNES_LUES = {
    "MONTH": "int8", "DAY": "int8", "DAY_OF_WEEK": "int8",
    "AIRLINE": "category", "FLIGHT_NUMBER": "int16",
    "ORIGIN_AIRPORT": "str", "DESTINATION_AIRPORT": "str",
    "SCHEDULED_DEPARTURE": "int16", "SCHEDULED_ARRIVAL": "int16",
    "SCHEDULED_TIME": "float32", "DISTANCE": "int16",
    "ARRIVAL_DELAY": "float32", "DIVERTED": "int8", "CANCELLED": "int8",
}

# Colonnes connues seulement pendant ou après le vol : interdites comme variables explicatives.
COLONNES_INTERDITES = [
    "DEPARTURE_TIME", "DEPARTURE_DELAY", "TAXI_OUT", "WHEELS_OFF", "ELAPSED_TIME",
    "AIR_TIME", "WHEELS_ON", "TAXI_IN", "ARRIVAL_TIME", "ARRIVAL_DELAY",
    "AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY",
    "WEATHER_DELAY", "CANCELLATION_REASON", "CANCELLED", "DIVERTED", "RETARD",
]

# Jours fériés fédéraux américains de 2015 (+ 1er janvier 2016 pour la fin d'année)
JOURS_FERIES = pd.to_datetime([
    "2015-01-01", "2015-01-19", "2015-02-16", "2015-05-25", "2015-07-04", "2015-09-07",
    "2015-10-12", "2015-11-11", "2015-11-26", "2015-12-25", "2016-01-01",
])


# ---------------------------------------------------------------------------
# 1. Chargement et nettoyage
# ---------------------------------------------------------------------------
def trouver_donnees(data_dir: str | None = None) -> str:
    """Renvoie le dossier contenant flights.csv.

    Ordre de recherche : `data_dir` s'il est fourni, puis le dossier `data/` du projet,
    puis téléchargement automatique avec kagglehub (mis en cache dans ~/.cache/kagglehub).
    """
    candidats = [data_dir] if data_dir else []
    candidats.append(os.path.join(os.path.dirname(__file__), "..", "data"))
    for d in candidats:
        if d and os.path.exists(os.path.join(d, "flights.csv")):
            return os.path.abspath(d)
    if data_dir:
        raise FileNotFoundError(f"flights.csv introuvable dans {data_dir}")
    import kagglehub
    return kagglehub.dataset_download("usdot/flight-delays")


def est_code_iata(s: pd.Series) -> pd.Series:
    """Vrai si le code d'aéroport est un code IATA à 3 lettres (ex. « JFK »)."""
    return s.str.fullmatch(r"[A-Z]{3}").fillna(False).astype(bool)


def construire_correspondance_octobre(df: pd.DataFrame, part_min: float = 0.9) -> dict[str, str]:
    """Reconstruit la correspondance « numéro d'aéroport → code IATA » pour octobre 2015.

    En octobre, le fichier code les aéroports par un identifiant numérique du DOT
    (ex. « 12478 » au lieu de « JFK »). On retrouve le code en appariant les vols réguliers
    d'octobre avec ceux de septembre et novembre : même compagnie, même numéro de vol,
    même heure prévue et même distance ⇒ mêmes aéroports. Chaque appariement est un « vote »
    ; on garde le code majoritaire s'il recueille au moins `part_min` des votes.
    """
    cle = ["AIRLINE", "FLIGHT_NUMBER", "SCHEDULED_DEPARTURE", "DISTANCE"]
    aero = ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
    numerique = ~est_code_iata(df["ORIGIN_AIRPORT"])
    octobre = df.loc[numerique & (df["MONTH"] == 10), cle + aero].drop_duplicates()
    reference = df.loc[~numerique & df["MONTH"].isin([9, 11]), cle + aero].drop_duplicates()
    paires = octobre.merge(reference, on=cle, suffixes=("_num", "_iata"))

    votes = pd.concat([
        paires[[f"{c}_num", f"{c}_iata"]].set_axis(["num", "iata"], axis=1) for c in aero
    ])
    comptes = votes.value_counts().rename("n").reset_index()
    comptes["part"] = comptes["n"] / comptes.groupby("num")["n"].transform("sum")
    meilleurs = comptes.sort_values("n").groupby("num").tail(1)
    meilleurs = meilleurs[meilleurs["part"] >= part_min]
    # Un code IATA ne doit correspondre qu'à un seul numéro : on écarte les ambiguïtés
    meilleurs = meilleurs[~meilleurs["iata"].duplicated(keep=False)]
    return dict(zip(meilleurs["num"], meilleurs["iata"]))


def corriger_codes_octobre(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Remplace les codes numériques d'octobre par les codes IATA et retire les vols non appariés."""
    corresp = construire_correspondance_octobre(df)
    for col in ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]:
        df[col] = df[col].replace(corresp)
    ok = est_code_iata(df["ORIGIN_AIRPORT"]) & est_code_iata(df["DESTINATION_AIRPORT"])
    return df[ok].copy(), corresp


def ajouter_trafic(df: pd.DataFrame) -> pd.DataFrame:
    """Nombre de départs **prévus** depuis le même aéroport, le même jour, dans la même heure.

    Calculé sur le programme complet (vols annulés compris, car ils étaient prévus),
    **avant** l'échantillonnage : sinon le trafic serait sous-estimé et bruité.
    Les horaires étant publiés des mois à l'avance, cette variable n'est pas une fuite.
    """
    heure = (df["SCHEDULED_DEPARTURE"] // 100).astype("int8")
    df["TRAFIC_HEURE"] = (
        df.groupby([df["ORIGIN_AIRPORT"], df["MONTH"], df["DAY"], heure], observed=True)["MONTH"]
        .transform("size").astype("int16")
    )
    return df


def charger_vols(data_dir: str, sample_size: int | None = 1_500_000, seed: int = 42,
                 verbose: bool = True) -> pd.DataFrame:
    """Charge flights.csv, corrige octobre, calcule le trafic, nettoie et échantillonne.

    Renvoie un DataFrame trié chronologiquement avec la cible RETARD (1 = plus de 15 min).
    """
    df = pd.read_csv(os.path.join(data_dir, "flights.csv"), usecols=list(COLONNES_LUES),
                     dtype=COLONNES_LUES)
    n0 = len(df)

    df, corresp = corriger_codes_octobre(df)
    n_oct = n0 - len(df)

    df = ajouter_trafic(df)

    # Vols annulés ou déroutés : pas de retard à l'arrivée mesurable
    df = df[(df["CANCELLED"] == 0) & (df["DIVERTED"] == 0)]
    df = df.dropna(subset=["ARRIVAL_DELAY", "SCHEDULED_TIME"])
    df = df.drop(columns=["CANCELLED", "DIVERTED", "FLIGHT_NUMBER"])
    n_propres = len(df)

    df["RETARD"] = (df["ARRIVAL_DELAY"] > SEUIL_RETARD).astype("int8")
    for col in ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]:
        df[col] = df[col].astype("category")

    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=seed)
    df["DATE"] = pd.to_datetime(dict(year=2015, month=df["MONTH"], day=df["DAY"]))
    df = df.sort_values(["DATE", "SCHEDULED_DEPARTURE"]).reset_index(drop=True)
    df.attrs["correspondance_octobre"] = corresp  # gardée pour vérification dans le notebook

    if verbose:
        print(f"Vols dans le fichier           : {n0:>10,}")
        print(f"Codes d'octobre corrigés       : {len(corresp):>10} aéroports "
              f"({n_oct:,} vols non appariés retirés)")
        print(f"Après retrait annulés/déroutés : {n_propres:>10,}")
        print(f"Échantillon gardé              : {len(df):>10,}")
        print(f"Part de vols en retard         : {df['RETARD'].mean():>10.1%}")
    return df


# ---------------------------------------------------------------------------
# 2. Variables de calendrier et d'horaires
# ---------------------------------------------------------------------------
def ajouter_variables_calendrier(df: pd.DataFrame) -> pd.DataFrame:
    """Heures prévues, jour de l'année et distance (en jours) au jour férié le plus proche."""
    df["HEURE_DEP"] = (df["SCHEDULED_DEPARTURE"] // 100).astype("int8")
    df["HEURE_ARR"] = ((df["SCHEDULED_ARRIVAL"] // 100) % 24).astype("int8")
    df["JOUR_ANNEE"] = df["DATE"].dt.dayofyear.astype("int16")
    # Calcul sur les dates uniques (365) puis report : bien moins de mémoire qu'une matrice n × 11
    dates = pd.Series(df["DATE"].unique())
    ecart = np.abs(dates.values[:, None] - JOURS_FERIES.values[None, :]).min(axis=1)
    jours = pd.Series((ecart / np.timedelta64(1, "D")).astype("int16"), index=dates)
    df["JOURS_FERIE"] = df["DATE"].map(jours).astype("int16")
    return df


# ---------------------------------------------------------------------------
# 3. Taux de retard historiques (encodage par la cible, sans fuite)
# ---------------------------------------------------------------------------
# Pour chaque aéroport, compagnie et ligne, on calcule la part de vols en retard
# **dans le passé**. Deux pièges à éviter :
#   1. utiliser le résultat du vol lui-même ou de vols futurs → fuite de données ;
#   2. faire confiance à une catégorie vue 3 fois (« 2 retards sur 3 = 67 % ») → bruit.
# Le 2e point est traité par un lissage bayésien (m-estimate) :
#       taux_lissé = (retards + m × taux_global) / (vols + m)
# Avec peu de vols, on reste proche du taux global ; avec beaucoup, on suit les données.

GROUPES_HISTORIQUES = {
    "TAUX_HIST_ORIGINE": ["ORIGIN_AIRPORT"],
    "TAUX_HIST_DESTINATION": ["DESTINATION_AIRPORT"],
    "TAUX_HIST_COMPAGNIE": ["AIRLINE"],
    "TAUX_HIST_LIGNE": ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"],
}


def _taux_cumules_avant_le_jour(passe: pd.DataFrame, cles: list[str], m: float) -> pd.DataFrame:
    """Taux lissé par (clé, jour) calculé avec les vols **strictement antérieurs** à ce jour."""
    # Taux global du passé, jour par jour (sert d'a priori pour le lissage)
    glob = passe.groupby("DATE")["RETARD"].agg(["sum", "count"]).sort_index()
    glob_avant = glob.cumsum() - glob  # cumul exclusif : uniquement les jours précédents
    prior = (glob_avant["sum"] / glob_avant["count"].replace(0, np.nan)).rename("prior")

    jour = passe.groupby(cles + ["DATE"], observed=True)["RETARD"].agg(["sum", "count"]).sort_index()
    cumul = jour.groupby(level=cles, observed=True).cumsum() - jour  # exclut le jour même
    cumul = cumul.join(prior, on="DATE")
    cumul["taux"] = (cumul["sum"] + m * cumul["prior"]) / (cumul["count"] + m)
    return cumul[["taux"]].reset_index()


def taux_figes(passe: pd.DataFrame, cles: list[str], prior: float, m: float = 50) -> pd.DataFrame:
    """Taux lissé par catégorie, calculé sur tous les vols de `passe` (statistiques figées)."""
    stats = passe.groupby(cles, observed=True)["RETARD"].agg(["sum", "count"])
    stats["taux"] = (stats["sum"] + m * prior) / (stats["count"] + m)
    return stats[["taux"]].reset_index()


def ajouter_taux_historiques(df: pd.DataFrame, date_coupure, m: float = 50) -> pd.DataFrame:
    """Ajoute les colonnes TAUX_HIST_* sans aucune fuite de données.

    - Vols **avant** `date_coupure` (entraînement) : taux calculé sur les vols des jours
      **précédents** uniquement (fenêtre cumulée qui s'arrête la veille).
    - Vols **à partir de** `date_coupure` (validation / test) : taux **figé** à la date de
      coupure, calculé sur tout l'entraînement. Le modèle n'utilise donc jamais la cible des
      vols qu'il doit prédire, ni celle des vols situés entre la coupure et le départ.
    - Catégorie jamais vue : taux global (prior).
    """
    date_coupure = pd.Timestamp(date_coupure)
    df = df.copy()
    avant = df["DATE"] < date_coupure
    passe = df.loc[avant, ["DATE", "RETARD", "AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]]
    prior_fige = passe["RETARD"].mean()

    for nom, cles in GROUPES_HISTORIQUES.items():
        # a) Entraînement : fenêtre cumulée jusqu'à la veille
        cumul = _taux_cumules_avant_le_jour(passe, cles, m)
        valeurs_train = df.loc[avant, cles + ["DATE"]].merge(cumul, on=cles + ["DATE"], how="left")["taux"]

        # b) Validation / test : statistiques figées à la date de coupure
        stats = taux_figes(passe, cles, prior_fige, m)
        valeurs_futur = df.loc[~avant, cles].merge(stats, on=cles, how="left")["taux"]

        df[nom] = np.nan
        df.loc[avant, nom] = valeurs_train.to_numpy()
        df.loc[~avant, nom] = valeurs_futur.fillna(prior_fige).to_numpy()
        df[nom] = df[nom].astype("float32")
    return df


# ---------------------------------------------------------------------------
# 4. Séparation temporelle et variables du modèle
# ---------------------------------------------------------------------------
CAT = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
NUM_BASE = ["MONTH", "DAY_OF_WEEK", "JOUR_ANNEE", "HEURE_DEP", "HEURE_ARR", "JOURS_FERIE",
            "DISTANCE", "SCHEDULED_TIME", "TRAFIC_HEURE"]
NUM_HIST = list(GROUPES_HISTORIQUES)

MOIS_TRAIN, MOIS_TEST = range(1, 9), [9, 11, 12]
DEBUT_VALIDATION, DEBUT_TEST = "2015-08-01", "2015-09-01"


def separer_temps(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Masques booléens de la séparation temporelle (jamais aléatoire).

    La période d'entraînement va de janvier à août ; sa fin (août) sert de validation :
    - entrainement : janvier → juillet (apprentissage des paramètres)
    - validation   : août (arrêt anticipé, choix du seuil)
    - test         : septembre, novembre, décembre (octobre n'est pas utilisé pour le test)
    """
    d = df["DATE"]
    return {
        "entrainement": d < DEBUT_VALIDATION,
        "validation": (d >= DEBUT_VALIDATION) & (d < DEBUT_TEST),
        "test": df["MONTH"].isin(MOIS_TEST),
    }


def construire_jeu_modele(df: pd.DataFrame, m: float = 50) -> pd.DataFrame:
    """Ajoute les taux historiques et regroupe les aéroports rares, sans fuite.

    - entraînement et validation : taux avec coupure au 1er août (la validation ne voit que
      l'historique de janvier à juillet, comme le modèle) ;
    - test : taux figés au 1er septembre (au moment de prédire, on connaît aussi les retards
      d'août, mais rien de septembre ni après).
    """
    sep = separer_temps(df)
    jeu = ajouter_taux_historiques(df, DEBUT_VALIDATION, m)
    jeu.loc[sep["test"], NUM_HIST] = ajouter_taux_historiques(df, DEBUT_TEST, m).loc[sep["test"], NUM_HIST]
    return regrouper_aeroports_rares(jeu, sep["entrainement"])


def regrouper_aeroports_rares(df: pd.DataFrame, masque_train: pd.Series, top: int = 100) -> pd.DataFrame:
    """Regroupe les aéroports hors des `top` plus fréquents **de l'entraînement** dans « AUTRE »."""
    df = df.copy()
    for col in ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]:
        comptes = df.loc[masque_train, col].value_counts()
        garde = comptes[comptes > 0].head(top).index  # value_counts liste aussi les catégories absentes
        valeurs = df[col].astype(str).where(df[col].isin(garde), "AUTRE")
        df[col] = pd.Categorical(valeurs, categories=sorted(set(map(str, garde))) + ["AUTRE"])
    return df
