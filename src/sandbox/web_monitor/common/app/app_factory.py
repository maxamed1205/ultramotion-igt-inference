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
import logging

# from sandbox.web_monitor.common import app



# Import routes
from sandbox.web_monitor.common.app.api_routes import register_api_routes
from sandbox.web_monitor.common.app.ws_routes import register_ws_routes


def create_app(cfg):
    """Crée et retourne une instance complète de l’application FastAPI."""
    log = logging.getLogger("igt.dashboard")

    meta = cfg.get("dashboard", {})
    app = FastAPI( title="Ultramotion Web Monitor", version="0.2-dev", description="Monitoring temps réel du pipeline Ultramotion",)

    @app.middleware("http")
    async def verify_token(request: Request, call_next):
        # ⚠️ Important : les middlewares HTTP ne s'appliquent PAS aux WebSockets.
        # Donc inutile d'essayer de "return await call_next" pour elles.
        # Ce bloc ne gérera que les requêtes HTTP normales.
        if request.scope["type"] != "http":
            return await call_next(request)

        # Vérifie le jeton pour les requêtes HTTP classiques
        if meta.get("security", {}).get("enabled", False):
            expected = meta["security"].get("token")
            token = request.headers.get("X-API-Token")
            if token != expected:
                log.warning(f"Requête non autorisée depuis {request.client.host}")
                return JSONResponse({"error": "unauthorized"}, status_code=401)

        return await call_next(request)



    # Middleware CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # à restreindre (ex: ["http://192.168.*"])
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Gestion d’erreurs globale (pour éviter les traces brutes)
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request, exc):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc):
        return JSONResponse({"error": "invalid request", "details": exc.errors()}, status_code=422)

    # Montage statique
    app.mount("/assets", StaticFiles(directory="src/sandbox/web_monitor/assets"), name="assets")

    app.state.cfg = cfg  # accessible immédiatement par ws_routes

    # Enregistrement des routes (API et WebSocket)
    register_api_routes(app)
    register_ws_routes(app)

    # Log de démarrage
    log.info("Application FastAPI configurée avec CORS et sécurité LAN")


    #Route de santé (health check)
    @app.get("/api/health")
    async def health_check(request: Request):
        token = request.headers.get("X-API-Token")
        expected = meta.get("security", {}).get("token")
        if meta.get("security", {}).get("enabled", False) and token != expected:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return {"status": "ok"}

    # Ajout du router API
    # from sandbox.web_monitor.common import app
    from sandbox.web_monitor.common.app.api_routes import router as api_router
    app.include_router(api_router)

    log.info("Application FastAPI configurée avec CORS et sécurité LAN")
    app.state.cfg = cfg  # 🔹 Rendre la configuration accessible aux routes WS
    return app
