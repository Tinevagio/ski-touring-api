"""
Fonctions météo agrégées par lissage spatial sur N grilles proches.

Port direct de src/app.py de ski-touring-live. Aucune réécriture, pour
garantir que les scores sortis matchent exactement ceux du Streamlit.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from .data_loader import DataBundle


def get_weather_icon(meteo: dict) -> str:
    """Emoji météo selon conditions agrégées."""
    snow = meteo.get("total_snow", 0)
    temp = meteo.get("mean_temp", 0)
    precip = meteo.get("total_precip", 0)
    wind = meteo.get("max_wind", 0)

    if wind > 40:
        return "💨"
    if snow > 20:
        return "🌨️"
    elif snow > 5:
        return "🌨"
    if temp > 0 and precip > 5:
        return "🌧️"
    if temp > 0 and snow > 10:
        return "⚠️"
    if temp < -5 and snow < 2:
        return "☀️"
    if temp > 0:
        return "🌤️"
    return "⛅"


def get_physical_features(
    bundle: DataBundle,
    lat: float,
    lon: float,
    target_date: date,
    n_neighbors: int = 5,
) -> Optional[dict]:
    """Features 7j lissées sur N grilles voisines."""
    grid = bundle.grid_lookup
    df_meteo = bundle.df_meteo

    coords = grid[["latitude", "longitude"]].to_numpy()
    dists = np.sqrt((coords[:, 0] - lat) ** 2 + (coords[:, 1] - lon) ** 2)
    closest_indices = np.argsort(dists)[:n_neighbors]
    closest_grids = grid.iloc[closest_indices]
    closest_dists = dists[closest_indices]

    start_date = pd.to_datetime(target_date) - pd.Timedelta(days=7)
    end_date = pd.to_datetime(target_date)

    all_features: list[dict] = []
    weights: list[float] = []

    for idx, (_, g) in enumerate(closest_grids.iterrows()):
        mask = (
            (df_meteo["latitude"] == g["latitude"])
            & (df_meteo["longitude"] == g["longitude"])
            & (df_meteo["time"] > start_date)
            & (df_meteo["time"] <= end_date)
        )
        df_hist = df_meteo[mask]
        if df_hist.empty:
            continue
        t_min = df_hist["temperature_2m"].min()
        t_max = df_hist["temperature_2m"].max()
        features = {
            "temp_min_7d_avg": t_min,
            "temp_max_7d_avg": t_max,
            "temp_amp_7d_avg": t_max - t_min,
            "snowfall_7d_sum": df_hist["snowfall"].sum(),
            "wind_max_7d": df_hist["wind_speed_10m"].max(),
            "freeze_thaw_cycles_7d": int(
                ((df_hist["temperature_2m"].max() > 0)
                 & (df_hist["temperature_2m"].min() < 0)).sum()
            ),
        }
        all_features.append(features)
        weights.append(1.0 / (closest_dists[idx] + 0.01))

    if not all_features:
        return None

    w = np.array(weights)
    w = w / w.sum()

    return {
        "temp_min_7d_avg": float(sum(f["temp_min_7d_avg"] * wi for f, wi in zip(all_features, w))),
        "temp_max_7d_avg": float(sum(f["temp_max_7d_avg"] * wi for f, wi in zip(all_features, w))),
        "temp_amp_7d_avg": float(sum(f["temp_amp_7d_avg"] * wi for f, wi in zip(all_features, w))),
        "snowfall_7d_sum": float(sum(f["snowfall_7d_sum"] * wi for f, wi in zip(all_features, w))),
        "wind_max_7d":     float(max(f["wind_max_7d"] for f in all_features)),
        "freeze_thaw_cycles_7d": int(
            round(sum(f["freeze_thaw_cycles_7d"] * wi for f, wi in zip(all_features, w)))
        ),
    }


def get_meteo_agg(
    bundle: DataBundle,
    lat: float,
    lon: float,
    target_date: Optional[date] = None,
    n_neighbors: int = 3,
) -> dict:
    """Météo agrégée pour une journée donnée."""
    if target_date is None:
        from datetime import datetime
        target_date = datetime.today().date()

    grid = bundle.grid_lookup
    df_meteo = bundle.df_meteo

    coords = grid[["latitude", "longitude"]].to_numpy()
    dists = np.sqrt((coords[:, 0] - lat) ** 2 + (coords[:, 1] - lon) ** 2)
    closest_indices = np.argsort(dists)[:n_neighbors]
    closest_grids = grid.iloc[closest_indices]
    closest_dists = dists[closest_indices]

    all_meteo: list[dict] = []
    weights: list[float] = []

    for idx, (_, g) in enumerate(closest_grids.iterrows()):
        df_day = df_meteo[
            (df_meteo["latitude"] == g["latitude"])
            & (df_meteo["longitude"] == g["longitude"])
            & (df_meteo["time"].dt.date == target_date)
        ]
        # Fallback à la date disponible la plus proche
        if df_day.empty:
            df_grid = df_meteo[
                (df_meteo["latitude"] == g["latitude"])
                & (df_meteo["longitude"] == g["longitude"])
            ]
            if not df_grid.empty:
                df_grid_copy = df_grid.copy()
                df_grid_copy["date_diff"] = abs(
                    (df_grid_copy["time"].dt.date - target_date).apply(lambda x: x.days)
                )
                closest_date_idx = df_grid_copy["date_diff"].idxmin()
                closest_date = df_grid.loc[closest_date_idx, "time"].date()
                df_day = df_grid[df_grid["time"].dt.date == closest_date]
        if not df_day.empty:
            all_meteo.append({
                "mean_temp": float(df_day["temperature_2m"].mean()),
                "max_wind":  float(df_day["wind_speed_10m"].max()),
                "total_snow":  float(df_day["snowfall"].sum()),
                "total_precip": float(df_day["precipitation"].sum()),
            })
            weights.append(1.0 / (closest_dists[idx] + 0.01))

    if not all_meteo:
        return {
            "mean_temp": 0.0, "max_wind": 0.0, "total_snow": 0.0,
            "total_precip": 0.0, "data_available": False,
            "distance_km": float("inf"), "icon": "❓",
        }

    w = np.array(weights)
    w = w / w.sum()
    out = {
        "mean_temp":   float(sum(m["mean_temp"]   * wi for m, wi in zip(all_meteo, w))),
        "total_snow":  float(sum(m["total_snow"]  * wi for m, wi in zip(all_meteo, w))),
        "total_precip":float(sum(m["total_precip"]* wi for m, wi in zip(all_meteo, w))),
        "max_wind":    float(max(m["max_wind"] for m in all_meteo)),
        "data_available": True,
        "distance_km": float(closest_dists[0]),
    }
    out["icon"] = get_weather_icon(out)
    return out
