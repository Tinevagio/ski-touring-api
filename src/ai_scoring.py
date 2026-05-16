"""
Scoring IA hybride (LightGBM + règles métier).

Port direct de :
- compute_spring_snow_score
- compute_base_snow_score_boosted
- compute_hybrid_snow_score
- is_exceptional_winter_day / winter_exception_boost

Aucune réécriture logique.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from .data_loader import DataBundle


# ─── Helpers règles métier ──────────────────────────────────────────────────


def spring_activation_factor(snowfall_7d: float) -> float:
    if snowfall_7d <= 3:
        return 1.0
    elif snowfall_7d <= 10:
        return 0.5
    return 0.0


def freeze_quality(temp_min: float) -> float:
    if temp_min <= -6:
        return 1.0
    elif temp_min <= -3:
        return 0.8
    elif temp_min <= -1:
        return 0.6
    return 0.2


def thermal_amplitude_quality(temp_amp: float) -> float:
    if temp_amp >= 12:
        return 1.0
    elif temp_amp >= 8:
        return 0.8
    elif temp_amp >= 5:
        return 0.6
    return 0.3


def wind_penalty_spring(wind_max: float) -> float:
    if wind_max <= 15:
        return 1.0
    elif wind_max <= 30:
        return 0.7
    return 0.4


def is_exceptional_winter_day(features: dict) -> bool:
    return (
        features["snowfall_7d_sum"] >= 25
        and features["temp_min_7d_avg"] <= -6
        and features["wind_max_7d"] <= 35
    )


def winter_exception_boost(base_score: float, features: dict) -> float:
    if not is_exceptional_winter_day(features):
        return base_score
    headroom = 1.0 - base_score
    boosted = base_score + 0.5 * headroom
    return round(min(boosted, 1.0), 3)


# ─── Scores composés ────────────────────────────────────────────────────────


def compute_spring_snow_score(features: dict) -> float:
    activation = spring_activation_factor(features["snowfall_7d_sum"])
    if activation == 0:
        return 0.0
    freeze = freeze_quality(features["temp_min_7d_avg"])
    amp = thermal_amplitude_quality(features["temp_amp_7d_avg"])
    wind = wind_penalty_spring(features["wind_max_7d"])
    raw_score = 0.45 * freeze + 0.35 * amp + 0.20 * wind
    return round(raw_score * activation, 3)


def compute_base_snow_score_boosted(
    bundle: DataBundle, features: dict, date_sortie: date
) -> float:
    """Score hiver IA avec correction biais avalanche."""
    if bundle.ski_model is None:
        return 0.5

    input_data = pd.DataFrame([{
        "temp_min_7d_avg": features["temp_min_7d_avg"],
        "temp_max_7d_avg": features["temp_max_7d_avg"],
        "temp_amp_7d_avg": features["temp_amp_7d_avg"],
        "snowfall_7d_sum": features["snowfall_7d_sum"],
        "wind_max_7d": features["wind_max_7d"],
        "freeze_thaw_cycles_7d": features["freeze_thaw_cycles_7d"],
        "summit_altitude_clean": features.get("summit_altitude_clean", 2400),
        "topo_denivele": features.get("topo_denivele", 1200),
        "topo_difficulty": features.get("topo_difficulty", 3),
        "massif": features.get("massif", "MONT-BLANC"),
        "day_of_week": date_sortie.weekday(),
    }])
    input_data["massif"] = input_data["massif"].astype("category")

    score = bundle.ski_model.predict(input_data)[0]
    normalized = float(np.clip((score + 1) / 2, 0, 1))
    ml_boosted = 1 - (1 - normalized) ** 1.5
    final_score = winter_exception_boost(ml_boosted, features)
    final_score = final_score ** 0.65
    return round(final_score, 3)


def compute_hybrid_snow_score(
    bundle: DataBundle, features: dict, date_sortie: date
) -> tuple[float, float, float, str]:
    """
    Score hybride saisonnier.
    Retour : (hybrid, base, spring, saison_label)
    """
    month = date_sortie.month
    spring_score = compute_spring_snow_score(features)
    base_score = compute_base_snow_score_boosted(bundle, features, date_sortie)

    if month <= 2:
        return base_score, base_score, spring_score, "hiver"
    elif month == 3:
        day = date_sortie.day
        spring_weight = min(day / 31 * 0.6, 0.6)
        hybrid = (1 - spring_weight) * base_score + spring_weight * spring_score
        return float(hybrid), base_score, spring_score, "transition"
    elif 4 <= month <= 6:
        hybrid = max(spring_score, base_score * 0.7)
        return float(hybrid), base_score, spring_score, "printemps"
    else:
        return base_score, base_score, spring_score, "hiver"


def quality_label(note_10: float) -> tuple[str, str, str]:
    """Retourne (picto, qualité, color) selon la note sur 10."""
    if note_10 >= 8:
        return "⭐⭐⭐", "Excellente", "green"
    elif note_10 >= 6:
        return "⭐⭐", "Bonne", "blue"
    elif note_10 >= 4:
        return "⭐", "Moyenne", "orange"
    return "❄️", "Difficile", "red"
