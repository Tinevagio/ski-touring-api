"""
Chargement des données depuis le repo GitHub public de l'utilisateur.

Stratégie :
- Au démarrage, on télécharge les 4 fichiers depuis raw.githubusercontent.com
- On les garde en mémoire (DataFrames pandas + booster LightGBM)
- Cache TTL : 6h. Au-delà, on refetch automatiquement à la prochaine requête.

Le SHA n'est pas figé : on tire toujours de la branche `main` pour avoir la
version la plus fraîche (les CSV sont mis à jour quotidiennement par les
GitHub Actions cron du repo).
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Optional

import httpx
import lightgbm as lgb
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ─── URLs sources ───────────────────────────────────────────────────────────
# La branche est paramétrable via env var GITHUB_BRANCH (défaut: main).
# Pratique si on veut pointer sur une branche de test.

_REPO = "Tinevagio/Ski-touring-live"
_BRANCH = os.getenv("GITHUB_BRANCH", "main")
_BASE_URL = f"https://raw.githubusercontent.com/{_REPO}/{_BRANCH}"

URL_BERA      = f"{_BASE_URL}/data/bera_latest.csv"
URL_METEO     = f"{_BASE_URL}/data/meteo_cache.parquet"
URL_ITIN      = f"{_BASE_URL}/data/raw/itineraires_alpes_camptocamp.csv"
URL_MODEL     = f"{_BASE_URL}/models/skiability_regression_physical.txt"
URL_TRAIN_DS  = f"{_BASE_URL}/data/skitour_ml_dataset_openmeteo.csv"

# ─── Cache en mémoire ───────────────────────────────────────────────────────

_CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "21600"))  # 6h


@dataclass
class DataBundle:
    """Tout ce dont les endpoints ont besoin, chargé une fois et partagé."""
    df_itin: pd.DataFrame           # itinéraires Camptocamp + Skitour
    df_bera: pd.DataFrame           # BERA brut (date_validite, risque_actuel, ...)
    dict_bera: dict[str, float]     # massif → risque normalisé [0-1]
    df_meteo: pd.DataFrame          # météo horaire toutes grilles
    grid_lookup: pd.DataFrame       # latitudes/longitudes uniques (pour kNN)
    ski_model: Optional[lgb.Booster]
    # Catégories de massifs exactement dans l'ordre du training, pour aligner
    # pd.Categorical lors de l'inférence LightGBM (sinon "categorical_feature
    # do not match").
    massif_categories: list[str]
    # Mapping CSV-normalisé → nom training. Ex: "BEAUFORTAIN" → "Beaufortain"
    massif_csv_to_train: dict[str, str]
    fetched_at: float               # timestamp de chargement


_bundle: Optional[DataBundle] = None
_lock = threading.Lock()


def get_bundle() -> DataBundle:
    """
    Retourne le bundle de données. Refetch si on dépasse le TTL.

    Thread-safe : si plusieurs requêtes arrivent en même temps après expiration,
    une seule fetch sera lancée.
    """
    global _bundle
    now = time.time()
    if _bundle is not None and (now - _bundle.fetched_at) < _CACHE_TTL_SECONDS:
        return _bundle

    with _lock:
        # Re-check après acquisition du lock (un autre thread peut avoir
        # rafraîchi entre-temps)
        if _bundle is not None and (now - _bundle.fetched_at) < _CACHE_TTL_SECONDS:
            return _bundle
        log.info("Loading data bundle from GitHub (TTL expired or first load)…")
        _bundle = _load_from_remote()
        # Invalide les caches dérivés en aval (météo agrégée par cellule).
        try:
            from .meteo import clear_meteo_caches
            clear_meteo_caches()
        except Exception:
            pass
        log.info(
            "Bundle loaded: %d itinéraires, %d massifs BERA, %d points météo",
            len(_bundle.df_itin),
            len(_bundle.dict_bera),
            len(_bundle.df_meteo),
        )
        return _bundle


def force_reload() -> DataBundle:
    """Force un reload, ignore le TTL. Utile pour un endpoint admin /reload."""
    global _bundle
    with _lock:
        _bundle = _load_from_remote()
        # Invalide les caches dérivés (météo / features par cellule)
        try:
            from .meteo import clear_meteo_caches
            clear_meteo_caches()
        except Exception:
            pass
        return _bundle


# ─── Fetch HTTP ──────────────────────────────────────────────────────────────


def _http_get(url: str, timeout: float = 60.0) -> bytes:
    """GET avec retry léger. Render free tier peut avoir des hoquets réseau."""
    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            r = httpx.get(url, timeout=timeout, follow_redirects=True)
            r.raise_for_status()
            return r.content
        except httpx.HTTPError as e:
            last_exc = e
            log.warning("HTTP retry %d for %s: %s", attempt + 1, url, e)
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url} after 3 attempts: {last_exc}")


def _load_from_remote() -> DataBundle:
    """Télécharge et parse les 4 fichiers."""
    # ── BERA ────────────────────────────────────────────────────────────────
    bera_bytes = _http_get(URL_BERA)
    df_bera = pd.read_csv(io.BytesIO(bera_bytes))
    df_bera["massif"] = df_bera["massif"].astype(str).str.strip().str.upper()
    df_bera["date_validite"] = pd.to_datetime(
        df_bera["date_validite"], format="ISO8601", errors="coerce"
    )
    dict_bera = dict(
        zip(df_bera["massif"], df_bera["risque_actuel"].astype(float) / 5.0)
    )

    # ── Météo ───────────────────────────────────────────────────────────────
    meteo_bytes = _http_get(URL_METEO)
    df_meteo = pd.read_parquet(io.BytesIO(meteo_bytes))
    # On normalise la colonne `time` au cas où
    if not pd.api.types.is_datetime64_any_dtype(df_meteo["time"]):
        df_meteo["time"] = pd.to_datetime(df_meteo["time"], errors="coerce")
    unique_grids = (
        df_meteo[["latitude", "longitude"]]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # ── Itinéraires ─────────────────────────────────────────────────────────
    itin_bytes = _http_get(URL_ITIN)
    try:
        df_itin = pd.read_csv(io.BytesIO(itin_bytes), encoding="utf-8")
    except UnicodeDecodeError:
        df_itin = pd.read_csv(io.BytesIO(itin_bytes), encoding="cp1252")
    df_itin["massif"] = df_itin["massif"].astype(str).str.strip().str.upper()
    for col in ("lat", "lon", "denivele_positif"):
        df_itin[col] = pd.to_numeric(df_itin[col], errors="coerce")
    df_itin = df_itin.dropna(subset=["lat", "lon", "denivele_positif"]).reset_index(drop=True)

    # ── Modèle LightGBM ─────────────────────────────────────────────────────
    # lgb.Booster ne lit que depuis un fichier sur disque, pas depuis bytes.
    # On écrit dans un fichier temporaire avant de le charger.
    ski_model: Optional[lgb.Booster] = None
    try:
        model_bytes = _http_get(URL_MODEL)
        with tempfile.NamedTemporaryFile(
            suffix=".txt", delete=False, mode="wb"
        ) as fp:
            fp.write(model_bytes)
            tmp_path = fp.name
        try:
            ski_model = lgb.Booster(model_file=tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception as e:
        log.warning("Failed to load LightGBM model, AI scoring disabled: %s", e)

    # ── Catégories massif du training (pour aligner LightGBM) ──────────────
    # On fetch le dataset d'entraînement uniquement pour récupérer la liste
    # exacte des massifs et leur ordre d'apparition — c'est ce que LightGBM
    # attend en inférence. Sans ça, les noms en majuscules du CSV
    # itinéraires (BEAUFORTAIN) ne matchent pas avec ceux du training
    # (Beaufortain), et le score IA tombe à NaN.
    massif_categories: list[str] = []
    massif_csv_to_train: dict[str, str] = {}
    try:
        train_bytes = _http_get(URL_TRAIN_DS)
        df_train = pd.read_csv(io.BytesIO(train_bytes), usecols=["massif"])
        # Ordre d'apparition exact, sans dédoublonnage du DataFrame (drop_duplicates
        # conserve l'ordre d'apparition).
        massif_categories = df_train["massif"].drop_duplicates().tolist()
        # Mapping CSV → nom training. On filtre les massifs hors France
        # (Norvège, Argentine, etc.) qui n'ont pas de clé dans le mapping.
        for train_name in massif_categories:
            key = _normalize_massif(train_name)
            if key:
                massif_csv_to_train[key] = train_name
        log.info(
            "Loaded %d massif categories from training dataset (%d in CSV mapping)",
            len(massif_categories), len(massif_csv_to_train),
        )
    except Exception as e:
        log.warning("Failed to load training dataset for AI categories: %s", e)

    return DataBundle(
        df_itin=df_itin,
        df_bera=df_bera,
        dict_bera=dict_bera,
        df_meteo=df_meteo,
        grid_lookup=unique_grids,
        ski_model=ski_model,
        massif_categories=massif_categories,
        massif_csv_to_train=massif_csv_to_train,
        fetched_at=time.time(),
    )


def _normalize_massif(name: str) -> str:
    """
    Normalise un nom de massif pour mapping CSV ↔ training.
    On utilise un mapping explicite parce que la correspondance n'est pas
    mécanique (ex: 'Bornes - Aravis' côté training = 'ARAVIS' côté CSV).
    """
    # Mapping explicite : nom training → nom CSV. On inverse en sortie.
    train_to_csv = {
        'Vanoise': 'VANOISE',
        'Belledonne': 'BELLEDONNE',
        'Mont Blanc': 'MONT-BLANC',
        'Ecrins': 'ECRINS',
        'Vosges': 'VOSGES',
        'Alpes Grées S': 'ALPES-GREES-S',
        'Alpes Grées N': 'ALPES-GREES-N',
        'Grandes Rousses - Arves': 'GRANDES-ROUSSES',
        'Ubaye - Parpaillon - Alpes Cozie S': 'UBAYE',
        'Queyras - Alpes Cozie N': 'QUEYRAS',
        'Cerces - Thabor - Mont Cenis': 'THABOR',
        'Haut Giffre - Aiguilles Rouges': 'HAUT-GIFFRE',
        'Valais E - Alpes Pennines E': 'VALAIS-E',
        'Valais W - Alpes Pennines W': 'VALAIS-W',
        'Bornes - Aravis': 'ARAVIS',
        'Beaufortain': 'BEAUFORTAIN',
        'Chartreuse': 'CHARTREUSE',
        'Andorre - Ariège': 'ANDORRE',
        'Grand Paradis': 'GRAND-PARADIS',
        'Préalpes de Digne': 'PREALPES-DE-DIGNE',
        'Mercantour - Alpes Maritimes Italiennes': 'MERCANTOUR',
        'Taillefer - Matheysine': 'TAILLEFER',
        'Lauzière - Cheval Noir': 'LAUZIERE',
        'Préalpes de provence': 'PREALPES-DE-PROVENCE',
        'Devoluy': 'DEVOLUY',
        'Bauges': 'BAUGES',
        'Chablais - Faucigny': 'CHABLAIS',
        'Vercors': 'VERCORS',
        # Massifs hors France (très peu d'itinéraires CSV concernés)
        'Mont-Perdu - Gavarnie - Bigorre': 'MONT-PERDU',
        'Préalpes Fribourgeoises / Bernoises': 'PREALPES-FRIBOURGEOISES',
        'Cerdagne - Capcir - Conflent': 'CERDAGNE',
        'Luchon - Posets - Maladeta': 'LUCHON',
        'Alpes Bernoises E': 'ALPES-BERNOISES-E',
        'Alpes Bernoises W': 'ALPES-BERNOISES-W',
        'Alpes Vaudoises': 'ALPES-VAUDOISES',
        'Engadine': 'ENGADINE',
        'Silvretta': 'SILVRETTA',
        'Tessin E - Adula - Grisons': 'TESSIN-E',
        'Tessin W - Conches': 'TESSIN-W',
        'Alpes Glaronaises': 'ALPES-GLARONAISES',
        'Ortles': 'ORTLES',
        'Hohe Tauern': 'HOHE-TAUERN',
        'Stubai': 'STUBAI',
        'Ötztal': 'OETZTAL',
        'Vorarlberg Allgau': 'VORARLBERG',
        'Massif Central': 'MASSIF-CENTRAL',
        'Préalpes de Grasse': 'PREALPES-DE-GRASSE',
    }
    return train_to_csv.get(name, "")
