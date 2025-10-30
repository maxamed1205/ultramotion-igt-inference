"""
common/app/utils.py
--------------------
Outils communs pour le backend du Web Monitor :
- Réponses JSON standardisées
- Cache mémoire à durée courte
- Chargement des templates HTML
"""

import time

from datetime import datetime

import logging
from functools import lru_cache
from fastapi.responses import JSONResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path


# ─────────────────────────────────────────────
#  Réponses JSON uniformisées
# ─────────────────────────────────────────────

def json_ok(data=None, message="ok", status=200):
    """Retourne une réponse JSON standardisée pour les requêtes réussies."""
    payload = {
        "status": message,
        "timestamp": time.time(),
        "datetime": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "data": data or {}
    }
    return JSONResponse(payload, status_code=status)

def json_error(message="error", status=400, detail=None):
    """Retourne une réponse JSON standardisée pour les erreurs."""
    payload = {
        "status": "error",
        "message": message,
        "detail": detail,
        "timestamp": time.time(),
        "datetime": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    }
    return JSONResponse(payload, status_code=status)


# ─────────────────────────────────────────────
# Cache mémoire à courte durée
# ─────────────────────────────────────────────
_cache_ts = 0
_cache_data = None



def cached_snapshot(fetch_fn, ttl=1.0):
    """
    Exécute `fetch_fn()` (ex. lecture logs) mais met en cache le résultat
    pendant `ttl` secondes.
    """
    global _cache_ts, _cache_data
    now = time.time()
    if _cache_data is None or (now - _cache_ts) > ttl:
        try:
            _cache_data = fetch_fn()
            _cache_ts = now
            logging.getLogger("igt.dashboard").info(
                f"Cache refreshed at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
            )
        except Exception as e:
            logging.getLogger("igt.dashboard").warning(f"Cache snapshot error: {e}")
    return _cache_data


# ─────────────────────────────────────────────
# Chargement de templates HTML
# ─────────────────────────────────────────────
def load_templates():
    """
    Initialise l'environnement Jinja2 pour les templates HTML.
    Renvoie l'objet Environment prêt à être utilisé.
    """
    base_dir = Path(__file__).resolve().parents[2] / "templates"
    env = Environment(
        loader=FileSystemLoader(str(base_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    return env
