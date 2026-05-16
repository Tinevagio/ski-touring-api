"""
Scoring fitness/danger d'un itinéraire.

Port direct de scoring_v3() de src/app.py. Aucune modif logique.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from .data_loader import DataBundle
from .meteo import get_meteo_agg


def scoring_v3(
    bundle: DataBundle,
    row: pd.Series,
    niveau: str,
    dplus_min: int,
    dplus_max: int,
    target_date: date,
) -> float:
    """
    Score fitness/danger d'un itinéraire pour les conditions données.
    Plus haut = mieux.
    """
    # ── BERA ────────────────────────────────────────────────────────────────
    massif_key = row["massif"]
    avy_risk = bundle.dict_bera.get(massif_key, 0.6)  # défaut 3/5

    # ── Météo ──────────────────────────────────────────────────────────────
    meteo = get_meteo_agg(bundle, row["lat"], row["lon"], target_date)
    fresh_snow_penalty = min(meteo["total_snow"] / 30.0, 1.0)
    wet_snow_penalty = 1.0 if (meteo["mean_temp"] > 0 and meteo["total_precip"] > 0) else 0.0
    wind_penalty = min(meteo["max_wind"] / 25.0, 1.0)

    # ── Exposition & pente ─────────────────────────────────────────────────
    expo_map = {
        "N": 0.1, "NE": 0.2, "E": 0.4, "SE": 0.7,
        "S": 1.0, "SO": 0.8, "O": 0.6, "NO": 0.3, "NW": 0.2,
    }
    expo_penalty = expo_map.get(
        str(row["exposition"]).strip().upper()[:2], 0.5
    )
    slope_penalty = (
        1.0
        if str(row["difficulty_ski"]).strip().upper().startswith(("S4", "S5"))
        else 0.3
    )

    # ── Danger global ──────────────────────────────────────────────────────
    danger = (
        0.30 * avy_risk
        + 0.20 * wind_penalty
        + 0.15 * fresh_snow_penalty
        + 0.15 * wet_snow_penalty
        + 0.10 * expo_penalty
        + 0.10 * slope_penalty
    )

    # ── Fitness ────────────────────────────────────────────────────────────
    diff_ski = str(row["difficulty_ski"]).strip().upper()
    user_level = niveau.strip().upper()
    level_order = {"S1": 1, "S2": 2, "S3": 3, "S4": 4, "S5": 5}
    route_level = next(
        (v for k, v in level_order.items() if diff_ski.startswith(k)), 3
    )
    target_level = level_order.get(user_level, 3)
    level_diff = abs(route_level - target_level)
    level_bonus = 1.0 / (1 + level_diff)

    try:
        dplus = float(row["denivele_positif"])
    except Exception:
        dplus = 1000

    if dplus_min <= dplus <= dplus_max:
        range_center = (dplus_min + dplus_max) / 2
        range_width = max(dplus_max - dplus_min, 1)
        distance_from_center = abs(dplus - range_center) / range_width
        dplus_bonus = 1.0 - (0.3 * distance_from_center)
    else:
        if dplus < dplus_min:
            dplus_bonus = max(0.1, dplus / max(dplus_min, 1) * 0.5)
        else:
            dplus_bonus = max(0.1, dplus_max / max(dplus, 1) * 0.5)

    fitness = dplus_bonus * level_bonus

    return float(fitness / (1 + danger))
