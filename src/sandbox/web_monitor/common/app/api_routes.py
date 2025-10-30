"""
common/app/api_routes.py
-------------------------
Déclare les routes REST (API HTTP) du Web Monitor.
Phase 2.2A : version minimale, testable sans dépendances externes.
"""


from fastapi import APIRouter
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import time

from .utils import json_ok, cached_snapshot

# Routeur FastAPI (permet de modulariser le code)
router = APIRouter(prefix="/api/v1")

# ─────────────────────────────────────────────
# /api/health — état du service
# ─────────────────────────────────────────────
@router.get("/health")
async def health():
    """Retourne un statut simple pour vérifier que le service fonctionne."""
    return json_ok({
        "service": "Ultramotion Web Monitor"
    })



# ─────────────────────────────────────────────
# /api/metrics — exemple de données simulées
# ─────────────────────────────────────────────

def _fetch_mock_metrics():
    return {
        "gpu": {"usage": 73.2, "vram_used": 1456, "temp": 55.3},
        "cpu": {"usage": 21.8, "threads": 18},
        "fps": {"rx": 25.0, "proc": 24.8, "tx": 24.9},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

@router.get("/metrics")
async def get_metrics():
    data = cached_snapshot(_fetch_mock_metrics, ttl=1.0)
    return json_ok(data)


# ─────────────────────────────────────────────
# api/frames — placeholder (pas encore de pipeline)
# ─────────────────────────────────────────────
@router.get("/frames")
async def get_frames():
    """
    Retourne une liste simulée de frames disponibles.
    Plus tard : infos sur images, masks, etc.
    """
    frames = [
        {"id": 1, "name": "frame_00157.png", "timestamp": "2025-10-30 09:00:01"},
        {"id": 2, "name": "frame_00158.png", "timestamp": "2025-10-30 09:00:03"},
    ]
    return json_ok({"frames": frames})

from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.responses import RedirectResponse, HTMLResponse
from pathlib import Path

def register_api_routes(app):
    """Enregistre les routes API et frontend statiques."""
    app.include_router(router)

    base_dir = Path(__file__).resolve().parents[2]
    templates = Jinja2Templates(directory=str(base_dir / "templates"))

    # 🏠 Redirection racine
    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/dashboard")

    # 📊 Page principale rendue via Jinja2
    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard(request: Request):
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,  # ⚠️ obligatoire pour FastAPI
                "title": "Dashboard Temps Réel | UltraMotion IGT"
            }
        )


    # Note: Les fichiers statiques sont maintenant montés dans app_factory.py
