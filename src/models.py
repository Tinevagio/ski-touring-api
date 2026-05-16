"""Schémas Pydantic des réponses HTTP."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class MeteoSummary(BaseModel):
    icon: str
    mean_temp: float
    total_snow: float
    max_wind: float
    total_precip: float


class BeraSummary(BaseModel):
    risque: Optional[int]
    risque_color: Optional[str]


class FeaturesDetail(BaseModel):
    temp_min_7d_avg: float
    temp_max_7d_avg: float
    temp_amp_7d_avg: float
    snowfall_7d_sum: float
    wind_max_7d: float
    freeze_thaw_cycles_7d: int
    spring_score: float
    base_score: float


class Idea(BaseModel):
    name: str
    massif: str
    lat: float
    lon: float
    denivele_positif: float
    exposition: str
    difficulty_ski: str
    url: Optional[str] = None
    source: Optional[str] = None

    # Score principal (scoring_v3, sert au tri)
    score: float

    # Score IA hybride (Excellente / Bonne / Moyenne / Difficile)
    ai_snow_score: Optional[float] = None
    ai_note_10: Optional[float] = None
    ai_qualite: Optional[str] = None
    ai_picto: Optional[str] = None
    ai_color: Optional[str] = None
    ai_saison_mode: Optional[str] = None  # hiver / transition / printemps

    meteo: MeteoSummary
    bera: BeraSummary
    features_detail: Optional[FeaturesDetail] = None


class IdeasResponse(BaseModel):
    date: str
    saison: str
    weather_icon: str
    weather_alerts: list[str]
    ideas: list[Idea]
    stats: dict


class Metadata(BaseModel):
    massifs: list[str]
    dates_available: list[str]
    meteo_latest: Optional[str]
    bera_latest: Optional[str]
    nb_itineraires: int


class HealthResponse(BaseModel):
    status: str = Field("ok")
    data_loaded: bool
    fetched_at: Optional[float]
    cache_age_seconds: Optional[float]
    nb_itineraires: int = 0
    nb_massifs_bera: int = 0
    nb_grid_points: int = 0
    ai_model_available: bool = False
