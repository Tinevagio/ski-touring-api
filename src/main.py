"""
Ski Touring Live — API FastAPI.

Endpoints :
  - GET /health          : status + état du cache
  - GET /metadata        : massifs disponibles, dates dispo, fraîcheur
  - GET /ideas           : top N itinéraires selon filtres utilisateur
  - POST /admin/reload   : force le reload des données (utile en dev)

Le service est conçu pour fonctionner sur Render free tier. Cold start ~30s
(boot Python + uvicorn) + ~15s (fetch GitHub) + ~10s (chargement LightGBM
et parquet) = ~55s au pire. Le client Flutter ping /health au démarrage du
module pour réveiller le service pendant que l'utilisateur configure ses
filtres.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from . import ai_scoring
from .data_loader import force_reload, get_bundle
from .meteo import get_meteo_agg, get_weather_icon
from .models import (
    BeraSummary,
    FeaturesDetail,
    HealthResponse,
    Idea,
    IdeasResponse,
    MeteoSummary,
    Metadata,
)
from .scoring import scoring_v3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ski-touring-api")

app = FastAPI(
    title="Ski Touring Live API",
    version="1.0.0",
    description="API de recommandation d'itinéraires de ski de rando.",
)

# CORS permissif : ce service est consommé par l'app mobile WhiteSilence
# (origines diverses selon le contexte). Pas de données sensibles servies.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Handler global pour les 500 : on log le traceback complet côté serveur
# (sinon Render n'affiche qu'un "Internal Server Error" générique côté client
# sans info utile). Important pour le diagnostic.
@app.exception_handler(Exception)
async def _unhandled(request, exc):
    import traceback
    from fastapi.responses import JSONResponse
    tb = traceback.format_exc()
    log.error("Unhandled exception on %s %s:\n%s",
              request.method, request.url.path, tb)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal error: {type(exc).__name__}: {exc}"},
    )


# ─── /health ─────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Health check. Si le bundle n'est pas encore chargé, on ne le charge PAS
    ici (pour ne pas faire payer 30s à un simple ping). Le chargement se fait
    lazy à la première requête /ideas.
    """
    from . import data_loader
    b = data_loader._bundle
    if b is None:
        return HealthResponse(
            status="warming",
            data_loaded=False,
            fetched_at=None,
            cache_age_seconds=None,
        )
    return HealthResponse(
        status="ok",
        data_loaded=True,
        fetched_at=b.fetched_at,
        cache_age_seconds=time.time() - b.fetched_at,
        nb_itineraires=len(b.df_itin),
        nb_massifs_bera=len(b.dict_bera),
        nb_grid_points=len(b.grid_lookup),
        ai_model_available=b.ski_model is not None,
    )


@app.post("/admin/reload", response_model=HealthResponse)
def admin_reload() -> HealthResponse:
    """Force un reload immédiat depuis GitHub. Pas d'auth pour l'instant."""
    b = force_reload()
    return HealthResponse(
        status="ok",
        data_loaded=True,
        fetched_at=b.fetched_at,
        cache_age_seconds=0.0,
        nb_itineraires=len(b.df_itin),
        nb_massifs_bera=len(b.dict_bera),
        nb_grid_points=len(b.grid_lookup),
        ai_model_available=b.ski_model is not None,
    )


# ─── /metadata ───────────────────────────────────────────────────────────────


@app.get("/metadata", response_model=Metadata)
def metadata() -> Metadata:
    """Liste des massifs et dates dispo pour remplir les listes côté Flutter."""
    b = get_bundle()
    massifs = sorted(b.df_itin["massif"].unique().tolist())

    today = datetime.today().date()
    all_dates = sorted(b.df_meteo["time"].dropna().dt.date.unique())
    dates_futures = [d.isoformat() for d in all_dates if d >= today][:4]

    meteo_latest = None
    if not b.df_meteo.empty:
        meteo_latest = str(b.df_meteo["time"].max().date())

    bera_latest = None
    if not b.df_bera.empty:
        bd = b.df_bera["date_validite"].max()
        if bd is not None and not isinstance(bd, float):  # pas NaT
            bera_latest = str(bd)

    return Metadata(
        massifs=massifs,
        dates_available=dates_futures,
        meteo_latest=meteo_latest,
        bera_latest=bera_latest,
        nb_itineraires=len(b.df_itin),
    )


# ─── /ideas ──────────────────────────────────────────────────────────────────


def _risque_color_emoji(risque: int) -> str:
    colors = ["🟢", "🟡", "🟠", "🔴", "⚫"]
    if 1 <= risque <= 5:
        return colors[risque - 1]
    return "⚪"


def _parse_list_param(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [s.strip().upper() for s in value.split(",") if s.strip()]


@app.get("/ideas", response_model=IdeasResponse)
def ideas(
    date: str = Query(..., description="ISO date (YYYY-MM-DD)"),
    niveau: str = Query("S3", regex="^S[1-5]$"),
    dplus_min: int = Query(800, ge=0, le=3000),
    dplus_max: int = Query(1500, ge=0, le=3000),
    expositions: str = Query("N,NE,E,SE,S,SO,O,NO", description="liste séparée par virgules"),
    massifs: Optional[str] = Query(None, description="liste séparée par virgules. None = tous"),
    n_results: int = Query(5, ge=1, le=50),
    include_ai: bool = Query(True, description="inclut le score IA hybride (plus lent)"),
) -> IdeasResponse:
    """
    Trouve les meilleurs itinéraires pour les conditions données.

    Mêmes critères que le formulaire Streamlit. Identique en logique à
    `scoring_v3()` du repo ski-touring-live.
    """
    # ── Validation ─────────────────────────────────────────────────────────
    try:
        target_date = datetime.fromisoformat(date).date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Date invalide (attendu YYYY-MM-DD)")

    if dplus_min > dplus_max:
        raise HTTPException(status_code=400, detail="dplus_min > dplus_max")

    expositions_list = _parse_list_param(expositions)
    if not expositions_list:
        raise HTTPException(status_code=400, detail="Au moins une exposition")

    massifs_list = _parse_list_param(massifs) if massifs else None

    bundle = get_bundle()

    # ── Vérif fraîcheur météo (comme dans Streamlit) ───────────────────────
    if not bundle.df_meteo.empty:
        meteo_latest = bundle.df_meteo["time"].max().date()
        days_old = (target_date - meteo_latest).days
        if days_old > 3:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Données météo trop anciennes pour le {target_date}. "
                    f"Dernière météo : {meteo_latest}."
                ),
            )

    # ── Filtrage ───────────────────────────────────────────────────────────
    t0 = time.time()
    df = bundle.df_itin
    df_filtered = df[
        df["denivele_positif"].between(dplus_min, dplus_max)
        & df["exposition"].astype(str).str.strip().str.upper().isin(expositions_list)
    ].copy()
    if massifs_list:
        df_filtered = df_filtered[df_filtered["massif"].isin(massifs_list)]

    n_filtered = len(df_filtered)
    log.info("[perf] /ideas: %d itinéraires filtrés en %.2fs",
             n_filtered, time.time() - t0)
    if n_filtered == 0:
        return IdeasResponse(
            date=target_date.isoformat(),
            saison=_saison_label(target_date),
            weather_icon="❓",
            weather_alerts=[],
            ideas=[],
            stats={
                "n_filtered_before_score": 0,
                "meteo_latest": str(meteo_latest) if not bundle.df_meteo.empty else None,
                "bera_latest": None,
                "cache_age_seconds": time.time() - bundle.fetched_at,
            },
        )

    # ── Scoring v3 (fitness/danger) ────────────────────────────────────────
    t0 = time.time()
    df_filtered["score"] = df_filtered.apply(
        lambda row: scoring_v3(bundle, row, niveau, dplus_min, dplus_max, target_date),
        axis=1,
    )
    log.info("[perf] /ideas: scoring_v3 sur %d lignes en %.2fs",
             n_filtered, time.time() - t0)
    top = df_filtered.sort_values("score", ascending=False).head(n_results).copy()

    # ── Météo globale + alertes de la journée ──────────────────────────────
    df_jour = bundle.df_meteo[bundle.df_meteo["time"].dt.date == target_date]
    alerts: list[str] = []
    weather_icon_global = "⛅"
    if not df_jour.empty:
        mean_snow = float(df_jour["snowfall"].mean())
        mean_temp = float(df_jour["temperature_2m"].mean())
        max_wind = float(df_jour["wind_speed_10m"].max())
        weather_icon_global = get_weather_icon({
            "total_snow": mean_snow,
            "mean_temp": mean_temp,
            "total_precip": float(df_jour["precipitation"].mean()),
            "max_wind": max_wind,
        })
        if mean_snow > 20:
            alerts.append("❄️ Neige fraîche abondante (20+ cm) → Risque plaques à vent")
        if mean_temp > 0:
            alerts.append("☀️ Températures positives → Éviter expositions Sud (coulées)")
        if max_wind > 40:
            alerts.append("💨 Vent fort (40+ km/h) → Attention aux crêtes")

    # ── Construction de la réponse ─────────────────────────────────────────
    t0 = time.time()
    ideas_out: list[Idea] = []
    for _, row in top.iterrows():
        meteo = get_meteo_agg(bundle, row["lat"], row["lon"], target_date)

        # BERA
        massif_key = row["massif"]
        risque_int: Optional[int] = None
        if massif_key in bundle.dict_bera and not bundle.df_bera.empty:
            bera_row = bundle.df_bera[bundle.df_bera["massif"] == massif_key]
            if not bera_row.empty:
                risque_int = int(bera_row.iloc[0]["risque_actuel"])
        bera_sum = BeraSummary(
            risque=risque_int,
            risque_color=_risque_color_emoji(risque_int) if risque_int else None,
        )

        # Score IA hybride (optionnel)
        ai_payload: dict = {}
        features_detail: Optional[FeaturesDetail] = None
        if include_ai and bundle.ski_model is not None:
            from .meteo import get_physical_features
            try:
                feat = get_physical_features(bundle, row["lat"], row["lon"], target_date)
                if feat is not None:
                    # pd.Series.get() existe mais retourne pd.NA sur clé absente,
                    # ce qui fait planter float(). On lit avec un fallback safe.
                    alt = _safe_float(row, "alt_sommet", default=2500.0)
                    feat["summit_altitude_clean"] = alt
                    feat["topo_denivele"] = _safe_float(row, "denivele_positif", default=1200.0)
                    feat["topo_difficulty"] = 3
                    feat["massif"] = str(row["massif"])
                    hybrid, base, spring, saison = ai_scoring.compute_hybrid_snow_score(
                        bundle, feat, target_date
                    )
                    note_10 = round(hybrid * 10, 1)
                    picto, qualite, color = ai_scoring.quality_label(note_10)
                    ai_payload = {
                        "ai_snow_score": float(hybrid),
                        "ai_note_10": float(note_10),
                        "ai_qualite": qualite,
                        "ai_picto": picto,
                        "ai_color": color,
                        "ai_saison_mode": saison,
                    }
                    features_detail = FeaturesDetail(
                        temp_min_7d_avg=feat["temp_min_7d_avg"],
                        temp_max_7d_avg=feat["temp_max_7d_avg"],
                        temp_amp_7d_avg=feat["temp_amp_7d_avg"],
                        snowfall_7d_sum=feat["snowfall_7d_sum"],
                        wind_max_7d=feat["wind_max_7d"],
                        freeze_thaw_cycles_7d=int(feat["freeze_thaw_cycles_7d"]),
                        spring_score=float(spring),
                        base_score=float(base),
                    )
            except Exception as e:
                # On ne casse pas la requête si l'IA plante sur un itinéraire.
                # On log et on continue : la card sera affichée sans score IA.
                log.warning(
                    "[ideas] AI scoring failed for %s (%s): %s",
                    row.get("name", "?"), row.get("massif", "?"), e,
                )
                ai_payload = {}
                features_detail = None

        ideas_out.append(Idea(
            name=str(row["name"]),
            massif=str(row["massif"]),
            lat=float(row["lat"]),
            lon=float(row["lon"]),
            denivele_positif=float(row["denivele_positif"]),
            exposition=str(row["exposition"]),
            difficulty_ski=str(row["difficulty_ski"]),
            url=str(row["url"]) if "url" in row and not _is_null(row["url"]) else None,
            source=str(row["source"]) if "source" in row and not _is_null(row["source"]) else None,
            score=float(row["score"]),
            meteo=MeteoSummary(
                icon=meteo["icon"],
                mean_temp=float(meteo["mean_temp"]),
                total_snow=float(meteo["total_snow"]),
                max_wind=float(meteo["max_wind"]),
                total_precip=float(meteo["total_precip"]),
            ),
            bera=bera_sum,
            features_detail=features_detail,
            **ai_payload,
        ))

    log.info("[perf] /ideas: construction réponse (n=%d, ai=%s) en %.2fs",
             len(ideas_out), include_ai, time.time() - t0)

    return IdeasResponse(
        date=target_date.isoformat(),
        saison=_saison_label(target_date),
        weather_icon=weather_icon_global,
        weather_alerts=alerts,
        ideas=ideas_out,
        stats={
            "n_filtered_before_score": n_filtered,
            "meteo_latest": str(bundle.df_meteo["time"].max().date()) if not bundle.df_meteo.empty else None,
            "bera_latest": str(bundle.df_bera["date_validite"].max()) if not bundle.df_bera.empty else None,
            "cache_age_seconds": time.time() - bundle.fetched_at,
        },
    )


def _saison_label(d: date) -> str:
    m = d.month
    if m <= 2:
        return "hiver"
    elif m == 3:
        return "transition"
    elif 4 <= m <= 6:
        return "printemps"
    return "hiver"


def _is_null(v) -> bool:
    import pandas as pd
    try:
        return pd.isna(v)
    except (TypeError, ValueError):
        return v is None


def _safe_float(row, col: str, *, default: float) -> float:
    """
    Lit row[col] en float avec fallback robuste.
    - Si la colonne n'existe pas : retourne default.
    - Si la valeur est NaN/None/non castable : retourne default.
    `row` peut être un dict ou une pd.Series.
    """
    try:
        if col not in row:
            return default
        v = row[col]
    except Exception:
        return default
    if _is_null(v):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ─── Lancement local / fallback ──────────────────────────────────────────────
# Permet de démarrer le service avec `python -m src.main` en cas de problème
# avec la start command. Force host=0.0.0.0 (obligatoire sur Render) et lit
# le port depuis l'env $PORT (avec fallback 8000 pour le local).
if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    log.info("Starting uvicorn on 0.0.0.0:%d", port)
    uvicorn.run("src.main:app", host="0.0.0.0", port=port, reload=False)
