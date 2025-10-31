"""
Module : app_factory.py
-----------------------
Crée et configure l'application FastAPI pour le Web Monitor :
- CORS pour accès LAN
- sécurité via X-API-Token
- montage statiques (/assets)
- enregistrement routes REST et WebSocket
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pathlib import Path
import logging

# Collector réel : LogCollector lit les fichiers pipeline.log et kpi.log
from sandbox.web_monitor.common.collector.log_collector.collector import LogCollector as Collector

# Import des routes
from sandbox.web_monitor.common.app.api_routes import register_api_routes
from sandbox.web_monitor.common.app.ws_routes import register_ws_routes


def create_app(cfg):
    """Crée et retourne une instance complète de l’application FastAPI."""
    log = logging.getLogger("igt.dashboard")

    meta = cfg.get("dashboard", {})
    app = FastAPI(
        title="Ultramotion Web Monitor",
        version="0.2-dev",
        description="Monitoring temps réel du pipeline Ultramotion",
    )

    # ------------------------------------------------------------------
    # 🔒 Middleware de vérification du jeton HTTP (pas WebSocket)
    # ------------------------------------------------------------------
    @app.middleware("http")
    async def verify_token(request: Request, call_next):
        if request.scope["type"] != "http":
            return await call_next(request)

        if meta.get("security", {}).get("enabled", False):
            expected = meta["security"].get("token")
            token = request.headers.get("X-API-Token")
            if token != expected:
                log.warning(f"Requête non autorisée depuis {request.client.host}")
                return JSONResponse({"error": "unauthorized"}, status_code=401)

        return await call_next(request)

    # ------------------------------------------------------------------
    # 🌐 CORS + gestion d'erreurs globale
    # ------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # à restreindre plus tard si besoin
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request, exc):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc):
        return JSONResponse({"error": "invalid request", "details": exc.errors()}, status_code=422)

    # ------------------------------------------------------------------
    # 📦 Montage des fichiers statiques
    # ------------------------------------------------------------------
    base_dir = Path(__file__).resolve().parents[2]
    assets_path = base_dir / "assets"
    javascript_path = assets_path / "javascript"

    app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")
    app.mount("/javascript", StaticFiles(directory=str(javascript_path)), name="javascript")

    app.state.cfg = cfg

    # ------------------------------------------------------------------
    # 🧩 Collector : réel si disponible, sinon fallback simulé
    # ------------------------------------------------------------------
    try:
        app.state.collector = Collector()
        log.info("[APP] Collector réel attaché à app.state.collector ✅")
    except Exception as e:
        log.warning(f"[APP] Collector réel indisponible ({e}); utilisation du mock ⚠️")
        try:
            from sandbox.web_monitor.common.collector.log_collector.collector import LogCollector
            import tempfile

            tmp1 = tempfile.NamedTemporaryFile(delete=False).name
            tmp2 = tempfile.NamedTemporaryFile(delete=False).name
            app.state.collector = LogCollector(tmp1, tmp2)
            log.info("[APP] Collector simulé attaché à app.state.collector (fallback)")
        except Exception as e2:
            log.error(f"[APP] Impossible d’attacher un collector (réel ni simulé): {e2}")

    # ------------------------------------------------------------------
    # 🔌 Enregistrement des routes
    # ------------------------------------------------------------------
    register_api_routes(app)
    register_ws_routes(app)

    # ------------------------------------------------------------------
    # 💓 Health Check
    # ------------------------------------------------------------------
    @app.get("/api/health")
    async def health_check(request: Request):
        token = request.headers.get("X-API-Token")
        expected = meta.get("security", {}).get("token")
        if meta.get("security", {}).get("enabled", False) and token != expected:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return {"status": "ok"}

    from sandbox.web_monitor.common.app.api_routes import router as api_router
    app.include_router(api_router)

    log.info("Application FastAPI configurée avec CORS et sécurité LAN")
    return app
