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
