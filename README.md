# Ski Touring API

API FastAPI de recommandation d'itinéraires de ski de rando, alimentée par
les données du projet [Ski-touring-live](https://github.com/Tinevagio/Ski-touring-live).

C'est le backend du module **Idées** de WhiteSilence. Il porte la logique
de scoring de l'app Streamlit en API JSON consommable nativement par Flutter.

## Architecture

```
ski-touring-api/
├── src/
│   ├── main.py          # FastAPI app + routes
│   ├── data_loader.py   # fetch GitHub + cache TTL 6h
│   ├── meteo.py         # get_meteo_agg, get_physical_features
│   ├── scoring.py       # scoring_v3 (fitness/danger)
│   ├── ai_scoring.py    # LightGBM + hybride printemps/hiver
│   └── models.py        # schémas Pydantic
├── requirements.txt
└── render.yaml          # config déploiement Render
```

Les données (CSV BERA, météo Parquet, CSV itinéraires, modèle LightGBM)
sont **fetchées en HTTPS au démarrage** depuis le repo public
`Tinevagio/Ski-touring-live`, branche `main`. Cache 6h en mémoire.

## Endpoints

### `GET /health`
Status du service. Ne charge pas les données (fast).
```json
{ "status": "warming|ok", "data_loaded": true, ... }
```

### `GET /metadata`
Liste des massifs et dates disponibles. Sert à remplir les listes côté Flutter.
```json
{
  "massifs": ["ARAVIS", "BELLEDONNE", ...],
  "dates_available": ["2026-05-16", "2026-05-17", ...],
  "meteo_latest": "2026-05-19",
  "bera_latest": "2026-05-16",
  "nb_itineraires": 1716
}
```

### `GET /ideas`
Le cœur du métier. Retourne le top N selon les filtres.
```
GET /ideas?date=2026-05-18&niveau=S3&dplus_min=800&dplus_max=1500
   &expositions=N,NE,E,O,NO&massifs=ARAVIS,BELLEDONNE&n_results=5
```

Paramètres :
- `date` : ISO `YYYY-MM-DD`
- `niveau` : `S1`-`S5`
- `dplus_min`, `dplus_max` : dénivelé positif en mètres
- `expositions` : liste CSV (`N,NE,E,SE,S,SO,O,NO`)
- `massifs` : liste CSV optionnelle (vide = tous)
- `n_results` : 1-50
- `include_ai` : true/false. Si false, pas de score IA (plus rapide)

Réponse : voir `models.py` (classe `IdeasResponse`).

### `POST /admin/reload`
Force un reload depuis GitHub, ignore le TTL. Utile en dev ou après un push
manuel des données.

## Lancer en local

```bash
pip install -r requirements.txt
uvicorn src.main:app --reload
# → http://localhost:8000/docs (Swagger UI)
```

Premier appel à `/ideas` ou `/metadata` télécharge les données (~15s).

## Déploiement Render

1. Push ce repo sur GitHub
2. Sur Render → "New Web Service" → connecter le repo
3. Render détecte `render.yaml` automatiquement
4. Premier déploiement : ~3min (install des deps Python)
5. Cold start ensuite : ~30s

Variables d'env utiles :
- `GITHUB_BRANCH` : branche source (défaut `main`)
- `CACHE_TTL_SECONDS` : TTL du cache en mémoire (défaut 21600 = 6h)

## Conformité avec ski-touring-live

Le scoring (`scoring_v3`) et le modèle IA hybride (`compute_hybrid_snow_score`)
sont des **copies littérales** des fonctions de `src/app.py` de
ski-touring-live. Les résultats doivent être identiques pour des entrées
identiques.

Si tu modifies la logique de scoring dans ski-touring-live, il faut
synchroniser ici. À terme, on pourrait factoriser dans un package commun.
